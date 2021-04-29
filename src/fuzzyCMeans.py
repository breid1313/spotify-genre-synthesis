import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from preprocess import preprocess
from fcmeans import FCM
from jqm.jqmcvi import base
from functools import reduce


def secondLargestValue(list):
    maxValue = max(list[0], list[1])
    secondMax = min(list[0],list[1])
    n = len(list)
    for i in range(2, n):
        if list[i] > maxValue:
            secondMax = maxValue
            maxValue = list[i]
        elif list[i] > secondMax and maxValue != list[i]:
            secondMax = list[i]
    return secondMax

def calculateDistance(dataItem1, dataItem2):
    return np.sqrt(
        reduce(lambda x, y: x + y, [(dataItem1[i] - dataItem2[i]) ** 2 for i in range(len(dataItem1))])
    )


def FuzzyCMeans(dataFile, numComponents = 2, numClusters = 11, sampleSize = None):

    # read data into a DataFrame
    tracks = pd.read_csv(dataFile)
    # use a small subset
    if(sampleSize):
        tracks = tracks.head(sampleSize)

    X = preprocess(tracks, True).values
    pca = PCA(n_components = numComponents)

    X_PCA = pca.fit(X).transform(X)

    fuzzyCMean = FCM(n_clusters = numClusters)
    fuzzyCMean.fit(X_PCA)
    predicted = fuzzyCMean.predict(X_PCA)
    centers = fuzzyCMean.centers

    weights = fuzzyCMean.u
    secondaryLabels = list(map(lambda weightSet: np.where(weightSet == secondLargestValue(weightSet))[0][0], weights))

    closestToCenters = []
    for center in centers:
        distances = [] 
        for x in X_PCA:
            distances.append(calculateDistance(center, x))
        closestToCenters.append(X_PCA[distances.index(min(distances))])

    neighborSongs = []
    for center in closestToCenters:
        distances = [] 
        for x in X_PCA:
            if((center != x).all()):
                distances.append(calculateDistance(center, x))
        neighborSongs.append(X_PCA[distances.index(min(distances))])


    centerIndexes = list(map(lambda center: np.where(X_PCA == center)[0][0], closestToCenters))
    centerDataPointsXPca = np.array(list(map(lambda index: X_PCA[index], centerIndexes)))
    centerDataPointsRawData = list(map(lambda index: tracks.values[index], centerIndexes))
    secondaryCenterLabels = fuzzyCMean.predict(centerDataPointsXPca)

    neighborIndexes = list(map(lambda neighbor: np.where(X_PCA == neighbor)[0][0], neighborSongs))
    neighborsRawData = list(map(lambda index: tracks.values[index], neighborIndexes))

    colors = ['red', 'blue', 'yellow', 'purple', 'orange', 'green', 'black', 'pink', 'brown', 'gray', 'teal']
    labelColors = list(map(lambda index: colors[index], predicted))
    secondaryLabelColors = list(map(lambda index: colors[index], secondaryLabels))
    secondaryCenterLabelColors = list(map(lambda index: colors[index], secondaryCenterLabels))

    f = open("CenterSongs.txt", "a")
    for i in range(len(centerDataPointsRawData)):
        f.write(secondaryCenterLabelColors[i] + " Center:" + str(centerDataPointsRawData[i][1]) + ", " + str(centerDataPointsRawData[i][12]) + " Neighbor: " + str(neighborsRawData[i][1]) + ", " + str(neighborsRawData[i][12]) + '\n')
    f.close()

    silhouetteScore = silhouette_score(X_PCA, predicted)
    print(silhouetteScore)
    # code not optimized run at own risk
    # dunnIndex = base.dunn_fast(X_PCA, predicted)
    # print(dunnIndex)

    if(numComponents == 2):
        plt.scatter(X_PCA[:,0], X_PCA[:,1], c=labelColors, alpha=.8)
        plt.scatter(centers[:,0], centers[:,1], marker="+", s=500, c='w')

        plt.savefig('FCMeansWithPCA2' + str(numClusters) +'Clusters.png')


        plt.scatter(X_PCA[:,0], X_PCA[:,1], c=labelColors, alpha=.8)
        plt.scatter(centers[:,0], centers[:,1], marker="+", c="w", s=500)


        plt.scatter(X_PCA[:,0], X_PCA[:,1], c=secondaryLabelColors, alpha=.8)
        plt.scatter(centers[:,0], centers[:,1], marker="+", c=secondaryCenterLabelColors, s=500)
        plt.savefig('FCMeansWithPCA2' + str(numClusters) +'ClustersSecondary.png')
    elif(numComponents == 3):
        figure = plt.figure()
        scatterPlot = figure.add_subplot(projection='3d')
        scatterPlot.scatter(X_PCA[:,0], X_PCA[:,1], X_PCA[:,2], c=labelColors, alpha=.8)
        scatterPlot.scatter(centers[:,0], centers[:,1], centers[:,2], marker="+", s=500, c='w')

        plt.savefig('FCMeansWithPCA3' + str(numClusters) +'Clusters.png')