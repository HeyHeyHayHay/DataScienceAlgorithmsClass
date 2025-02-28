
import math
import numpy as np
import matplotlib.pyplot as plt

from kMeansClustering import kMeans
from kMeansClustering import plotClusters


# notes
# turn mininum in min to infinity

def averageInternalDistance(iPoint, cluster):

    averageInternalDistance = 0
    numberOfPointsInCluster = len(cluster)
    for point in cluster.values():
        averageInternalDistance += np.linalg.norm(iPoint - point)
        averageInternalDistance = averageInternalDistance / (numberOfPointsInCluster-1)


    return averageInternalDistance

def averageNeighborDistance(iPoint, neighborCluster):
    averageNeighborDistance = 0
    numberOfPointsInCluster = len(neighborCluster)
    for point in neighborCluster.values():
        averageNeighborDistance += np.linalg.norm(iPoint - point)
        averageNeighborDistance = averageNeighborDistance / numberOfPointsInCluster

    return averageNeighborDistance

def findNeighborClusters(iPointKeyValue, clusters):

    pointKey = iPointKeyValue[0]

    neighborClusters = clusters.copy()

    clusterOfPoint = findCluster(iPointKeyValue, clusters)

    neighborClusters.remove(clusterOfPoint)

    return neighborClusters

def findCluster(iPointKeyValue, clusters):
    pointKey = iPointKeyValue[0]
    clusterOfPoint = None

    for cluster in clusters:
        pointValue = cluster.get(pointKey)
        if (type(pointValue) != None):
            clusterOfPoint = cluster

    return clusterOfPoint

def minimumAverageNeighborDistance(iPointKeyValue, clusters):

    minimumClusterValue = 10000000000000
    neighborClusters = findNeighborClusters(iPointKeyValue, clusters)
    for cluster in neighborClusters:
        candidateDistance = averageNeighborDistance(iPointKeyValue[1], cluster)
        if candidateDistance < minimumClusterValue:
            minimumClusterValue = candidateDistance

    return minimumClusterValue

def pointSiCoefficient(iPointKeyValue, clusters):
    clusterOfPoint = findCluster(iPointKeyValue, clusters)
    numberOfPointsInClusterOfPoint = len(clusterOfPoint)
    iPointValue = iPointKeyValue[1]

    if numberOfPointsInClusterOfPoint > 1 :

        a = averageInternalDistance(iPointValue, clusterOfPoint)
        b = minimumAverageNeighborDistance(iPointValue, clusters)

        SiCoefficient = (b - a) / (max(a, b))
    else:
        SiCoefficient = 0

    return SiCoefficient

def SCoefficient(data, clusters):

    totalPoints = len(data)

    S = 0

    for pointKeyValue in data.items():
        S += pointSiCoefficient(pointKeyValue, clusters)

    S = S / totalPoints

    return S

def silhouetteKMeans(data, showFinal = False, plotSgraph = False, idPoints = False):

    totalPoints = len(data)
    k = 2
    maxIndex = 2
    maxSValue = -2

    SofK = []
    clustersOfK = []

    while ( (k - maxIndex < 4) and (k<totalPoints) and (k < 10000) ):

        clusters = kMeans(k, data)
        S = SCoefficient(data, clusters)
        SofK.append(S)
        clustersOfK.append(clusters)

        if (S > maxSValue):
            maxSValue = S
            maxIndex = k

        k = k + 1

    if plotSgraph:
        x = range(2,k)
        y = SofK

        plt.figure("Silhouette Coefficient")
        plt.plot(x,y)
        plt.show()


    print(maxIndex)
    bestClusters = clustersOfK[maxIndex-2]

    if showFinal:
        plotClusters(bestClusters, figureName = "Best Clustering", idPoints = False, show = True)

    return bestClusters

#testing
"""
numberOfDataPoints = 100
dimension = 2

randomData = {}
for i in range(numberOfDataPoints):
    randomData.update({f"{i}": np.random.rand(dimension)})

for i in range(numberOfDataPoints//2):
    value = randomData.get(f"{i}")

    value[0] += 2

    randomData.update({f"{i}": value})

for i in range(numberOfDataPoints//3):
    value = randomData.get(f"{i}")

    value[1] += 2

    randomData.update({f"{i}": value})

silhouetteKMeans(randomData, showFinal = True, plotSgraph = True)
"""
