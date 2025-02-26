import pandas as pd
import numpy as np
import kMeansClustering

zooDataFrame = pd.read_csv('datasets\zoo.csv')

zooDict = {}

for index, row in zooDataFrame.iterrows():
    array = zooDataFrame.iloc[index, 1:].to_numpy()
    zooDict.update({row['animal_name']: array})

classTaxon = kMeansClustering.kMeans(12, zooDict, plotFinal = True, idPoints = True)

# testing
for cluster in classTaxon:
    print(cluster.keys())
