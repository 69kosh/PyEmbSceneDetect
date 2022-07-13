from collections import namedtuple
import numpy as np


def calcScenes(distances: np.array, areas: np.array) -> tuple[dict, np.array]:
    '''
        попробую пойти снизу - сначала будет сцен столько сколько кадров
        итеративно, по минимальной стоимости объединения 
        (стоимость новой сцены размером с сумму старых), 
        объединяем сцены пока есть что объединять

        Ведем индекс по началу сцены и по концу сцены, а так же часть постоянно 
        сортируемого по цене объединения списка. обновляем эти списки при каждом слиянии
        помимо удаления влитого, обновления расширенного, еще пересчитываем обновления 
        с соседями

        TODO: нужно оптимизировать, очень прожорливый по памяти, долго выполняется
    '''

    Scene = namedtuple('Scene', ['b', 'e'])
    BiScene = namedtuple('BiScene', ['c', 's1', 's2'])

    length = distances.shape[0]
    windowSize = distances.shape[1]

    beginScenes = dict[int, Scene]()
    endScenes = dict[int, Scene]()
    unionCostScenes = list[BiScene]()
    # сюда попадают все объединения, у которых конец первой сцены или начало второй
    # чтобы при принятии объединения - удалить эти пересекающиеся предложения тоже
    s1EndBiScenes = dict[int, BiScene]()
    s2BeginBiScenes = dict[int, BiScene]()

    def getCost(i: int, s: int):
        return distances[i, s]/areas[i, s]

    def getDistanceArea(i: int, s: int):
        return (distances[i, s], areas[i, s])

    # заполняем
    for i in range(0, length - 1):

        s1 = Scene(i, i)
        s2 = Scene(i + 1, i + 1)
        bi = BiScene(getCost(s1.b, 1), s1, s2)
        s1bi = BiScene(getCost(i - 1, 1), Scene(i, i), s1)
        s2bi = BiScene(getCost(i + 1, 1), s2, Scene(i+2, i+2))

        beginScenes[s1.b] = s1
        endScenes[s1.e] = s1

        if i >= length - 2:
            beginScenes[s2.b] = s2
            endScenes[s2.e] = s2

        if i < length - 1:
            unionCostScenes.append(bi)

            if i < length - 2:
                s2BeginBiScenes[bi.s2.b] = bi

            if i > 0:
                s1EndBiScenes[bi.s1.e] = bi

    # unionCostScenes = sorted(unionCostScenes, key=lambda x: x[0])
    s = 0
    fullCost = 0
    fullSumDistances = 0
    fullSumAreas = len(beginScenes)
    costHistory = np.zeros(shape=(length))
    needSort = True
    # maxCost = 0
    allScenes = {}

    # сортируем вначале, потом только вставляем перед бОльшим весом
    unionCostScenes = sorted(unionCostScenes, key=lambda x: x.c)

    while len(unionCostScenes) > 1 and len(beginScenes) * windowSize > length:
        # while len (unionCostScenes) > 1 and len (beginScenes) > 100:
        s += 1
        # print(s)

        union = unionCostScenes.pop(0)

        # print('u', union)
        # удаляем два объединяющих предложения, пересекающиеся с выбранным
        s1Bi = s1EndBiScenes.get(union.s2.e, None)
        if s1Bi is not None:
            # print('lu',s1Bi)
            try:
                unionCostScenes.remove(s1Bi)
            except:
                # print(s1Bi)
                pass

        s2Bi = s2BeginBiScenes.get(union.s1.b, None)
        if s2Bi is not None:
            try:
                unionCostScenes.remove(s2Bi)
            except:
                # print(s2Bi)
                pass

        # считаем изменения (удаляемые сцены)
        (d1, a1) = getDistanceArea(union.s1.b, union.s1.e - union.s1.b)
        (d2, a2) = getDistanceArea(union.s2.b, union.s2.e - union.s2.b)
        fullSumDistances -= d1 + d2
        fullSumAreas -= a1 + a2

        # удаляем старые сцены
        del beginScenes[union.s1.b]
        del beginScenes[union.s2.b]
        del endScenes[union.s1.e]
        del endScenes[union.s2.e]

        # добавляем новую
        newScene = Scene(union.s1.b, union.s2.e)
        beginScenes[union.s1.b] = newScene
        endScenes[union.s2.e] = newScene

        # учитываем её
        (d, a) = getDistanceArea(newScene.b, newScene.e - newScene.b)
        fullSumDistances += d
        fullSumAreas += a

        # добавляем объединение с левой сценой и объединение с правой сценой
        # для этого находим левую и правую сцену

        leftScene = endScenes.get(union.s1.b - 1, None)
        if leftScene is not None and union.s2.e - leftScene.b < windowSize:
            # новый квадрат от начала левой сцены и заканчивая текущей
            bi = BiScene(getCost(leftScene.b, union.s2.e -
                         leftScene.b), leftScene, newScene)
            s1EndBiScenes[bi.s1.e] = bi
            s2BeginBiScenes[bi.s2.b] = bi
            unionCostScenes.append(bi)

        rightScene = beginScenes.get(union.s2.e + 1, None)
        if rightScene is not None and rightScene.e - union.s1.b < windowSize:
            # новый квадрат начиная от текущего и заканчивая концом правого
            bi = BiScene(getCost(union.s1.b, rightScene.e -
                         union.s1.b), newScene, rightScene)
            s1EndBiScenes[bi.s1.e] = bi
            s2BeginBiScenes[bi.s2.b] = bi
            unionCostScenes.append(bi)

        fullCost = fullSumDistances/fullSumAreas

        scenesLen = len(beginScenes)

        if (scenesLen > 10000 and scenesLen % 1000 == 0) \
                or (scenesLen < 10000 and scenesLen % 100 == 0) \
                or (scenesLen < 1000 and scenesLen % 10 == 0) \
                or (scenesLen < 100):
            unionCostScenes = sorted(unionCostScenes, key=lambda x: x.c)

        costHistory[scenesLen] = fullCost

        tempScenes = []
        for tuple in sorted(beginScenes.items()):
            tempScenes.append((tuple[1].b, tuple[1].e))

        if fullCost > 0.001 or scenesLen < length // 10:
            allScenes[scenesLen] = tempScenes

    return (allScenes, costHistory)


def chooseScenesProportion(allScenes: dict, allCosts: np.array, part=0.5) -> list:
    costHistoryProportionSum = np.sum(allCosts) * part
    costHistorySum = 0
    for i, cost in enumerate(allCosts):
        costHistorySum += cost
        if costHistorySum > costHistoryProportionSum:
            return allScenes[i]


def chooseScenesElbow(allScenes: dict, allCosts: np.array, rate=2) -> list:

    maxLenScenes = len(allCosts)

    maxCost = max(allCosts)
    costsFixed = [allCosts[j] + (j / maxLenScenes)*maxCost*rate for j in range(maxLenScenes)]

    minCost = 100500
    minScenes = []
    for i, cost in enumerate(costsFixed):
        if minCost > cost and i in allScenes:
            minScenes = allScenes[i]            
            minCost = cost
    return minScenes
