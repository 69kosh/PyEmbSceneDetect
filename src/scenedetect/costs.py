import numpy as np


def _calcCostsSquare(distances: np.array, offset: int = 0) -> tuple[np.array, np.array]:
    '''По матрице дистанций для одного кадра, всех размеров сцен считаем суммарные веса и площади
        Средняя стоимость, используемая дальше, это суммарная дистанция поделеная на площадь

        Дистанцию просчитываем для квадрата совпадающего с диагональю
        Подсчет ведётся итеративным обсчетом очередной грани (удвоенной ессесно)

        нулевой индекс соответствует единичной сцене, 
        соответственно всегда нулевая дистанция и единичная площадь

    '''
    length = distances.shape[0]
    windowSize = distances.shape[1]
    sums = np.zeros(shape=[windowSize], dtype=np.float32)

    sum = 0.0
    for size in range(1, windowSize):
        offset2 = offset + size
        if offset2 > length:
            # если выходим
            m = size/(length - offset)
            # тупо масштабируемся от последней суммы,
            # которая должна быть крайней на границе, ессесно квадратично
            sums[size] = sum * (m ** 2)
            # vector[size - 1] = sum
            # continue
            # возможно есть смысл выпилить, т.к. мы не должны использовать сцены,
            # заходящие своим окончание за границы
        else:
            # на каждый шаг добавляем к сумме
            # удвоенную сумму колонки (за диагональю, в конце квадрата)
            sum += np.sum(distances[offset: offset2 -
                          1, offset2 % windowSize]) * 2
            sums[size] = sum

    areas = np.arange(windowSize, dtype=np.float32) + 1
    areas = areas ** 2

    return (sums, areas)


def calcCostsSquare(distances: np.array) -> tuple[np.array, np.array]:
    '''считаем суммы дистанций и площади 
        для каждого кадра как начального, и для каждой длины сцены до размера окна

    '''
    length = distances.shape[0]
    windowSize = distances.shape[1]
    sums = np.zeros(shape=[length, windowSize], dtype=np.float32)
    areas = np.zeros(shape=[length, windowSize], dtype=np.float32)

    for i, v in enumerate(distances):
        # цена - суммарна дистанции деленная на площадь
        (sums[i], areas[i]) = _calcCostsSquare(
            distances=distances, offset=i)

    return (sums, areas)
