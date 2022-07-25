import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../src/')

from embscenedetect.scenes import calcScenes, chooseScenesElbow, chooseScenesProportion
from embscenedetect.costs import calcCostsSquare
from embscenedetect.distances import calcDistancesWindowed, normalizeDistances
from utils.video import prepareGeneratorFromVideo, generateDemoVideo, downloadVideoIfNeed
from utils.model import prepareModel, calcVectors
from utils.image import generateImage

filename = 'input.mp4'

every = 1

windowSize = 512

model = prepareModel()

downloadVideoIfNeed(
    url='https://www.youtube.com/watch?v=OMgIPnCnlbQ', filename=filename)

generator = prepareGeneratorFromVideo(filename=filename, every = every)

vectors = calcVectors(model=model, generator=generator)

print('Preparing distances...')

distances = calcDistancesWindowed(vectors=vectors, windowSize=windowSize)

distances = normalizeDistances(distances=distances)

generateImage(distances, 'distances.jpg')

print('Preparing costs...')

(sums, areas) = calcCostsSquare(distances=distances)
generateImage(sums/areas, 'costs.jpg')

print('Preparing scenes...')

(allScenes, allCosts) = calcScenes(distances=sums, areas=areas)

scenes = chooseScenesElbow(
    allScenes=allScenes, allCosts=allCosts, rate = 2)

print(scenes)

maxLenScenes = len(allCosts)

maxCost = max(allCosts)
costsFixed1 = [allCosts[j] + (j / maxLenScenes)*maxCost*1 for j in range(maxLenScenes)]
costsFixed2 = [allCosts[j] + (j / maxLenScenes)*maxCost*2 for j in range(maxLenScenes)]
costsFixed3 = [allCosts[j] + (j / maxLenScenes)*maxCost*3 for j in range(maxLenScenes)]

import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, len(allCosts), 0, np.amax(allCosts)])
plt.plot(allCosts)
plt.plot(costsFixed1)
plt.plot(costsFixed2)
plt.plot(costsFixed3)
plt.show()


generateDemoVideo(inputFilename=filename,
                  outputFilename='output.mp4', 
                  scenes=scenes, 
                  every = every)


from sklearn.cluster import AgglomerativeClustering

clustersCount = 10

clustering = AgglomerativeClustering(n_clusters=clustersCount).fit_predict(vectors)
clusterScenes = [[] for i in range(clustersCount)]

for scene in scenes:
    for k in range(scene[0], scene[1] + 1):
        cluster = clustering[k]
        if scene not in clusterScenes[cluster]:
            clusterScenes[cluster].append(scene)


for cluster, scenes in enumerate(clusterScenes):

    generateDemoVideo(inputFilename=filename,
                  outputFilename='output{}.mp4'.format(cluster), 
                  scenes=scenes, 
                  every = every)