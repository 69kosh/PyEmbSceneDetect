import numpy as np


def _calcCostsSquare(distances: np.array, offset: int = 0) -> tuple[np.array, np.array]:
    '''Вычисляем стоимость всех размеров сцены (от 0 до размера окна) 
        для конкретного начального кадра.

        Средняя стоимость, используемая дальше, это суммарная дистанция поделеная на площадь

        Дистанцию просчитываем для квадрата совпадающего с диагональю
        Подсчет ведётся итеративным обсчетом очередной грани (удвоенной)

        нулевой индекс соответствует единичной сцене, 
        соответственно всегда нулевая дистанция и единичная площадь
    '''
    length = distances.shape[0]
    windowSize = distances.shape[1]

    sums = np.zeros(shape=[windowSize], dtype=np.float32)
    areas = np.ones(shape=[windowSize], dtype=np.float32)

    sum = 0.0
    area = 1.0

    for size in range(1, windowSize):
        offset2 = offset + size
        if offset2 > length:
            continue

        # на каждый шаг добавляем к сумме
        # удвоенную сумму колонки (за диагональю, в конце квадрата)
        sum += np.sum(distances[offset: offset2, offset2 % windowSize]) * 2
        area += (offset2 - offset) * 2 + 1

        sums[size] = sum
        areas[size] = area

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


def _calcCostsHex(distances: np.array, distancesShifted: np.array, offset: int = 0,
                  minSplit: int = 8, maxSplit: int = 64, splitRate: float = 0.5) -> tuple[np.array, np.array]:
    '''Вычисляем стоимости всех размеров сцены (от 0 до размера окна) 
        для конкретного начального кадра.

        Средняя стоимость, используемая дальше, 
        это суммарная дистанция поделеная на площадь

        Дистанцию просчитываем для обрезанного квадрата совпадающего с диагональю
        Подсчет ведётся итеративным обсчетом очередной грани (удвоенной)

        Цель - сделать детектор более толерантным к плавным переходам, 
        когда дистанции меняются плавно из-за проводки камеры, 
        или действий в кадре

        .
         .
          .++--
          +.++-
          ++.++
          -++.+
          --++.
               .

        . - диагональные элементы
        + - то, что поадает к нам в выборку
        - - то, что не попадает к нам, но учитывалось в квадратной реализации

        Задаем минимальную границу отсеченя, максимальную границу отсечения и коэффициент отсечения, 
        например 0.5

        Для итеративного подсчета шестигранной дистанции используем 
        прямую матрицу дистанций для передней границы, 
        и сдвинутую матрицу для диагональной границы

    '''
    length = distances.shape[0]
    windowSize = distances.shape[1]
    sums = np.zeros(shape=[windowSize], dtype=np.float32)
    areas = np.ones(shape=[windowSize], dtype=np.float32)

    sum = 0.0
    area = 1.0
    oldSplitSize = minSplit
    for size in range(1, windowSize):
        # вычисляем высоту отсечения (с учетм минимального и максимального), если достигли
        # вычисляем дельту высоты отсечения от предыдущего шага
        # на каждом шаге берём колонку до высоты отсечения, если достигли
        # если дельта не нулевая - берём колонку у смешенной матрицы с данными границы отсечения
        '''
        0  v of+si % ws
        .             
         .
          .+++--        < of
           .+++-        < of+si
            .+++
             .++
              .+
               .
                .

        '''

        # это диагональные координаты выбираемой крайней колонки как по горизонту, так и по вертикали
        offset2 = offset + size

        # это размер от диагонали в минус, вверх
        # а для смешенных дистанций - это смещение диагонали слева
        splitSize = round(min(max(size * splitRate, minSplit), maxSplit))

        delta = abs(splitSize - oldSplitSize)
        oldSplitSize = splitSize

        # координата верхней границы колонки
        topOffset = offset2 - min(size, splitSize)

        if offset2 > length:
            continue

        # на каждый шаг добавляем к сумме
        # удвоенную сумму колонки (за диагональю, в конце квадрата), ограниченная сплитом
        sum += np.sum(distances[topOffset: offset2, offset2 % windowSize]) * 2
        area += (offset2 - topOffset) * 2 + 1

        # если дельта не нулевая - берём новую диагональную разделительную колонку
        if delta > 0:
            sum += np.sum(distancesShifted[offset: offset2 -
                          splitSize, splitSize]) * 2
            area += (offset2 - splitSize - offset) * 2

        sums[size] = sum

        areas[size] = area

    return (sums, areas)


def calcCostsHex(distances: np.array,
                 distancesShifted: np.array,
                 minSplit: int = 8,
                 maxSplit: int = 64,
                 splitRate: float = 0.5) -> tuple[np.array, np.array]:
    '''считаем суммы дистанций и площади 
        для каждого кадра как начального, и для каждой длины сцены до размера окна
        используем шестигранную область
    '''
    length = distances.shape[0]
    windowSize = distances.shape[1]
    sums = np.zeros(shape=[length, windowSize], dtype=np.float32)
    areas = np.zeros(shape=[length, windowSize], dtype=np.float32)

    for i, v in enumerate(distances):
        # цена - суммарна дистанции деленная на площадь
        (sums[i], areas[i]) = _calcCostsHex(
            distances=distances,
            distancesShifted=distancesShifted,
            offset=i,
            minSplit=minSplit,
            maxSplit=maxSplit,
            splitRate=splitRate)

    return (sums, areas)
