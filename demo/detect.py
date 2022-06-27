import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../src/')

from scenedetect.scenes import calcScenes, chooseScenesProportion
from scenedetect.costs import calcCostsSquare
from scenedetect.distances import calcDistancesWindowed, normalizeDistances
from utils.video import prepareGeneratorFromVideo, generateDemoVideo, downloadVideoIfNeed
from utils.model import prepareModel, calcVectors, normalizeVectors
from utils.image import generateImage


filename = 'input.mp4'

every = 1

windowSize = 256

model = prepareModel()

downloadVideoIfNeed(
    url='https://www.youtube.com/watch?v=sGbxmsDFVnE', filename=filename)

generator = prepareGeneratorFromVideo(filename=filename, every = every)

vectors = calcVectors(model=model, generator=generator)

vectors = normalizeVectors(vectors=vectors)

print('Preparing distances...')

distances = calcDistancesWindowed(vectors=vectors, windowSize=windowSize)

distances = normalizeDistances(distances=distances)

generateImage(distances, 'distances.jpg')

print('Preparing costs...')

(sums, areas) = calcCostsSquare(distances=distances)
generateImage(sums/areas, 'costs.jpg')

print('Preparing scenes...')

(allScenes, allCosts) = calcScenes(distances=sums, areas=areas)

scenes = chooseScenesProportion(
    allScenes=allScenes, allCosts=allCosts, part=0.2)

generateDemoVideo(inputFilename=filename,
                  outputFilename='output.mp4', 
                  scenes=scenes, 
                  every = every)

print(scenes)

import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, len(allCosts), 0, np.amax(allCosts)])
plt.plot(allCosts)
plt.show()


