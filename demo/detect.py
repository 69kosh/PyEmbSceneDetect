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

model = prepareModel() 

downloadVideoIfNeed(
    url='https://www.youtube.com/watch?v=OMgIPnCnlbQ', filename=filename)

generator = prepareGeneratorFromVideo(filename=filename, every = every)

vectors = calcVectors(model=model, generator=generator)

vectors = normalizeVectors(vectors=vectors)

distances = calcDistancesWindowed(vectors=vectors, windowSize=1024)

distances = normalizeDistances(distances=distances)

generateImage(distances, 'distances.jpg')

(sums, areas) = calcCostsSquare(distances=distances)
generateImage(sums/areas, 'costs.jpg')


(allScenes, allCosts) = calcScenes(distances=sums, areas=areas)

scenes = chooseScenesProportion(
    allScenes=allScenes, allCosts=allCosts, part=0.25)

print(scenes)

generateDemoVideo(inputFilename=filename,
                  outputFilename='output.mp4', 
                  scenes=scenes, 
                  every = every)
