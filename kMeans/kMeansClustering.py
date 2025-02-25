
import math
import random
import matplotlib.pyplot as plt
import numpy as np

# data

numberOfDataPoints = 1000
dimension = 2

randomData = {}
for i in range(numberOfDataPoints):
    randomData.update({f"{i}": np.random.rand(dimension)})

coordinatesOfColleges = {
    "MSU": np.array([42.726042170483964, -84.4790496844048]),  # Michigan State University
    "UTD": np.array([32.987360824843165, -96.74982117779595]),  # University of Texas at Dallas
    "Clemson": np.array([34.676477325642054, -82.84088117033222]),  # Clemson University
    "NCSU": np.array([35.78632005942455, -78.6812618067957]),  # North Carolina State University
    "UMN": np.array([44.97439085782415, -93.22820772065663]),  # University of Minnesota
    "UIUC": np.array([40.102625955846385, -88.22680396287784]),  # University of Illinois Urbana-Champaign
    "TAMU": np.array([30.621277067619435, -96.33441242127287]),  # Texas A&M University
    "UIowa": np.array([41.66370672513079, -91.55508087581431]),  # University of Iowa
    "ISU": np.array([42.027395455672206, -93.64447486872734]),  # Iowa State University
    "UDel": np.array([39.68187702767834, -75.75251399011542]),  # University of Delaware
    "WMU": np.array([42.28435667550862, -85.6108496736789])  # Western Michigan University
}

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

    clusters = [{} for i in range(len(means))]

    closestClusterIndices = []

    for id, point in data.items():
        closestClusterIndex = closestMeanIndex(means, point)
        clusters[closestClusterIndex].update({id: point})
        closestClusterIndices.append(closestClusterIndex)

    return clusters

def findCenterOfMass(cluster):
    numberOfDataPoints = len(cluster)
    clusterPoints = list(cluster.values())
    centerOfMass = clusterPoints[0].astype(np.float64)
    for point in clusterPoints[1:]:
        centerOfMass = centerOfMass + point.astype(np.float64)

    #print("numberOfDataPoints: ", numberOfDataPoints, "centerOfMass: ", centerOfMass, "cluster: ", cluster)

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
        for point in newClusters[k].values():
            inPrevious = sum(np.array_equal(point, comparisonPoint) for comparisonPoint in previousClusters[k].values())
            if (inPrevious == 0):
                indicationValue = indicationValue + 1

    return indicationValue

def initializeMeans(K, data):
    #randomize iniital points as means
    means = random.sample(list(data.values()), K)
    return means

def twoDPlotCluster(cluster, idPoints = False, show = False, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors=None, colorizer=None, plotnonfinite=False, data=None, **kwargs):

    x = [point[0] for point in cluster.values()]
    y = [point[1] for point in cluster.values()]

    plt.scatter(x, y, s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                alpha=alpha, linewidths=linewidths, edgecolors=edgecolors,
                plotnonfinite=plotnonfinite, data=data, **kwargs)

    if show:
        plt.show()

    if idPoints:
        keys = list(cluster.keys())
        for key, x_val, y_val in zip(keys, x, y):
            plt.annotate(str(key), (x_val, y_val), textcoords="offset points", xytext=(5,5), ha='right', fontsize=8)

    return

def twoDPlotAllClusters(clusters, idPoints = False, show = False, s=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors=None, colorizer=None, plotnonfinite=False, data=None, **kwargs):

    numberOfClusters = len(clusters)
    colors = plt.cm.tab10(np.linspace(0, 1, numberOfClusters))

    for i in range(numberOfClusters):
        twoDPlotCluster(clusters[i], idPoints, s=s, color=colors[i], marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, edgecolors=edgecolors, colorizer=colorizer, plotnonfinite=plotnonfinite, data=data, label=f'C {i+1}', **kwargs)

    if show:
        plt.legend()
        plt.show()

    return

def multiToTwoDCluster(cluster):
    twoDCluster = {}

    lengthOfVector = len(next(iter(cluster.values())))
    weight = np.random.rand(lengthOfVector)

    for key, value in cluster.items():
        lengthOfVector = len(value)
        half = lengthOfVector // 2
        weightedValue = np.multiply(weight, value)
        x = np.mean(weightedValue[:half])
        y = np.var(weightedValue[half:])
        twoDCluster[key] = np.array([x, y])
    return twoDCluster

# algorithm

def kMeans(K, data, plot = False, idPoints = False):

    if (K >= len(data)):
        raise ValueError("Too many clusters")

    #initialize means
    # choose first k elements of data. update to randomize

    previousMeans = initializeMeans(K, data)

    previousClusters = findClusters(previousMeans, data)

    dimensionOfPoints = len(next(iter(previousClusters[0].values())))

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
            plotClusters = []
            if dimensionOfPoints != 2:
                for cluster in newClusters:
                    plotCluster = multiToTwoDCluster(cluster)
                    plotClusters.append(plotCluster)
            else:
                for cluster in newClusters:
                    plotCluster = cluster
                    plotClusters.append(plotCluster)

            plt.figure(step)
            if idPoints:
                twoDPlotAllClusters(plotClusters, show = True, idPoints = True)
            else:
                twoDPlotAllClusters(plotClusters, show = True)

    means = newMeans

    return newClusters
