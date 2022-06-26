import numpy as np
from scipy.spatial.distance import cdist


def _calcRowDistances(vectors: np.array, vector: np.array) -> np.array:
    '''вычисляем дистанции между списком векторов и вектором
    '''
    return cdist(vectors, vector.reshape(1, -1), 'euclidean').reshape(-1)  # 'cosine', 'euclidean'


def calcDistancesWindowed(vectors: np.array, windowSize=1024, distanceFunc=_calcRowDistances) -> np.array:
    '''вычисляем и возвращаем двухмерный массив расстояний

        Для оптимизации обсчитывается только часть ограниченная окном
        и только для верхней левой части, нижнюю правую часть не считаем
        т.к. она зеркальна

        0, 0     0, windowSize
        .++++++++
        -.+++++++
        --.++++++
        ---.+++++
        ----.++++
        -----.+++
        ------.++
        ------ .+
        ------  .
        .+++++
         .++++
          .+++
           .++
            .+
             .
        n, 0

        где:
        . - нулевая диагональная дистанция
        + - то, правая верхняя часть дистанций
        - - правая верхняя часть дистанций, перенесенная к началу строки, 
            чтобы в пределах окна можно было пользоваться остатком от деления

        Это озволяет уйти от квадратичной сложности по вычислительным ресурсам,
        и что самое главное - по памяти

    '''
    length = len(vectors)

    distances = np.zeros(shape=[length, windowSize], dtype=np.float32)

    for i, v in enumerate(vectors):

        firstBeginDst = i % windowSize
        firstBeginSrc = 0

        # если текущий не в последнем куске
        if i < (length // windowSize) * windowSize:
            firstEndDst = windowSize
            firstEndSrc = windowSize - i % windowSize

        # последний кусок
        else:
            firstEndDst = length % windowSize
            firstEndSrc = length % windowSize - i % windowSize

        secondBeginDst = 0
        secondBeginSrc = windowSize - i % windowSize

        # если хвостом уткнулись в конец
        if i + windowSize <= length:
            secondEndDst = i % windowSize
            secondEndSrc = windowSize

        # если еще не начался последний кусок
        elif i < (length // windowSize) * windowSize:
            secondEndDst = length % windowSize
            secondEndSrc = (length - i) % windowSize

        # последний кусок
        else:
            secondEndDst = None
            secondEndSrc = None

        border = min(length, i + windowSize)
        distance = distanceFunc(vectors[i: border], v)  # от 0 до size

        distances[i, firstBeginDst: firstEndDst] = distance[firstBeginSrc: firstEndSrc]
        if secondEndDst is not None:
            distances[i, secondBeginDst: secondEndDst] = distance[secondBeginSrc: secondEndSrc]

    return distances


def calcDistancesWindowedShifted(vectors: np.array, windowSize=1024, distanceFunc=_calcRowDistances) -> np.array:
    """Вычисляем матрицу дистанций, ограниченную окном, диагональ смещена к началу оси
    
    """

    length = len(vectors)

    distances = np.zeros(shape=[length, windowSize], dtype=np.float32)

    for i, v in enumerate(vectors):

        border = min(length, i + windowSize)
        distance = distanceFunc(vectors[i: border], v)  # от 0 до size

        distances[i, 0: border - i] = distance

    return distances


def normalizeDistances(distances:np.array) -> np.array:

    distances /= np.max(np.abs(distances))

    return distances