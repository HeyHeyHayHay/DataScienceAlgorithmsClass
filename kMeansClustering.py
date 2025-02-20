
import math
import random
import matplotlib.pyplot as plt
import numpy as np

# data

numberOfDataPoints = 40
dimension = 2

data = []
for i in range(numberOfDataPoints):
    data.append(np.random.rand(dimension))

coordinatesOfColleges = [np.array([42.726042170483964, -84.4790496844048]), np.array([32.987360824843165, -96.74982117779595]), np.array([34.676477325642054, -82.84088117033222]), np.array([35.78632005942455, -78.6812618067957]), np.array([44.97439085782415, -93.22820772065663]), np.array([40.102625955846385, -88.22680396287784]), np.array([30.621277067619435, -96.33441242127287]), np.array([41.66370672513079, -91.55508087581431]), np.array([42.027395455672206, -93.64447486872734]), np.array([39.68187702767834, -75.75251399011542]), np.array([42.28435667550862, -85.6108496736789])]

# functions

def closestMeanIndex(means, point):
    minIndex = 0
    minDistance = math.inf
    for i in range(len(means)):
        mean = means[i]
        distance = np.linalg.norm(point - mean)
        if (distance == 0):
            return i
        elif (distance < minDistance):
            minDistance = distance
            minIndex = i

        index = minIndex
    return index

def findClusters(means, data):
    clusters = []
    for i in range(len(means)):
        clusters.append([])

    for point in data:
        closestClusterIndex = closestMeanIndex(means, point)
        clusters[closestClusterIndex].append(point)

    return clusters

def findCenterOfMass(cluster):
    numberOfDataPoints = len(cluster)
    centerOfMass = cluster[0]
    for i in range(1, numberOfDataPoints):
        centerOfMass = centerOfMass + cluster[i]

    centerOfMass = centerOfMass / numberOfDataPoints

    return centerOfMass

def updateAllCentersOfMass(clusters):
    means = []

    for cluster in clusters:
        means.append(findCenterOfMass(cluster))

    return means

def indicatorFunction(previousClusters, newClusters):

    indicationValue = 0

    K = len(previousClusters)

    for k in range(K):
        for point in newClusters[k]:
            inPrevious = sum(np.array_equal(point, previousClusters[k][i]) for i in range(len(previousClusters[k])))
            if (inPrevious == 0):
                indicationValue = indicationValue + 1

    return indicationValue

def initializeMeans(K, data):
    # choose first k elements of data. update to randomize

    means = []
    for i in range(K):
        means.append(data[i])

    return means

def twoDPlotCluster(cluster, show = False, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors=None, colorizer=None, plotnonfinite=False, data=None, **kwargs):

    x = [point[0] for point in cluster]
    y = [point[1] for point in cluster]

    plt.scatter(x, y, s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                alpha=alpha, linewidths=linewidths, edgecolors=edgecolors,
                plotnonfinite=plotnonfinite, data=data, **kwargs)

    if show:
        plt.show()

    return

def twoDPlotAllClusters(clusters, show = False, s=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors=None, colorizer=None, plotnonfinite=False, data=None, **kwargs):

    numberOfClusters = len(clusters)
    colors = plt.cm.tab10(np.linspace(0, 1, numberOfClusters))

    for i in range(numberOfClusters):
        twoDPlotCluster(clusters[i], s=s, color=colors[i], marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, edgecolors=edgecolors, colorizer=colorizer, plotnonfinite=plotnonfinite, data=data, label=f'C {i+1}', **kwargs)

    if show:
        plt.legend()
        plt.show()

    return

# algorithm

def kMeans(K, data, plot = False):

    if (K >= len(data)):
        raise ValueError("Too many clusters")

    #initialize means
    # choose first k elements of data. update to randomize

    previousMeans = initializeMeans(K, data)

    previousClusters = findClusters(previousMeans, data)

    newIndicator = math.inf
    previousIndicator = math.inf
    step = 0

    # repeat algorithm
    while ( (step < 10) or (previousIndicator > newIndicator) and (step < 1000)) and (newIndicator != 0):

        # update center of mass

        newMeans = updateAllCentersOfMass(previousClusters)

        # update clusters

        newClusters = findClusters(newMeans, data)

        # indicator
        previousIndicator = newIndicator

        newIndicator = indicatorFunction(previousClusters, newClusters)

        # update variables
        previousClusters = newClusters
        previousMeans = newMeans

        step = step + 1
        print("Step:", step)
        print("Indication Value: ", newIndicator)

        if plot:
            plt.figure(step)
            twoDPlotAllClusters(newClusters, show = True)

    means = newMeans

    return newClusters

# testing

kMeans(3, coordinatesOfColleges, plot = True)
