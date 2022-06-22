import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../src/')

from scenedetect.scenes import calcScenes, chooseScenesProportion
from scenedetect.costs import calcCostsSquare
from scenedetect.distances import calcDistancesWindowed
from utils.video import prepareGeneratorFromVideo, generateDemoVideo, downloadVideoIfNeed
from utils.model import prepareModel, calcVectors, normalizeVectors

import numpy as np
import cv2
import matplotlib.pyplot as plt

filename = 'input.mp4'

model = prepareModel()

downloadVideoIfNeed(
    url='https://www.youtube.com/watch?v=OMgIPnCnlbQ', filename=filename)

generator = prepareGeneratorFromVideo(filename=filename)

vectors = calcVectors(model=model, generator=generator)

vectors = normalizeVectors(vectors=vectors)

distances = calcDistancesWindowed(vectors=vectors)

(sums, areas) = calcCostsSquare(distances=distances)

(allScenes, allCosts) = calcScenes(distances=sums, areas=areas)

scenes = chooseScenesProportion(
    allScenes=allScenes, allCosts=allCosts, part=0.3)

generateDemoVideo(inputFilename=filename,
                  outputFilename='output.mp4', scenes=scenes)

print(scenes)
