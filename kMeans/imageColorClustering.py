from kMeansClustering import kMeans
from kMeansClustering import findCenterOfMass

import math
import random
import numpy as np
from PIL import Image


# data

def imageToDict(imageArray):

    height, width, _ = imageArray.shape
    pixelDict = {}

    for y in range(height):
        for x in range(width):
            pixelDict[(x, y)] = imageArray[y, x]

    return pixelDict

def clustersToImage(clusters, imageArray):

    originalImageHeight, originalImageWidth, _ = imageArray.shape

    newImageArray = np.zeros((originalImageHeight, originalImageWidth, 3), dtype=np.uint8)
    for cluster in clusters:
        averageRGB = np.clip(findCenterOfMass(cluster), 0, 255).astype(np.uint8)
        for (x, y) in cluster.keys():
            newImageArray[y, x] = averageRGB

    return Image.fromarray(newImageArray)

def clusterImageColors(numberOfColors, imagePath, plot = False, plotFinal = True, idPoints = False):

    image = Image.open(imagePath).convert("RGB")
    imageArray = np.array(image)

    height, width, _ = imageArray.shape

    imagePixelData = imageToDict(imageArray)
    clusters = kMeans(numberOfColors, imagePixelData, plot, plotFinal, idPoints)

    newImage = clustersToImage(clusters, imageArray)

    newImage.show()

    return newImage

reducedColorImage = clusterImageColors(12, "images/Paul_Cézanne,_Still_Life_1890.jpg", plot = False, plotFinal = True)
