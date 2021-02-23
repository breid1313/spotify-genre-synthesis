import numpy as np
import random

def KMeans(kClusters, numIterations, attributes, data):
    clusters = []
    for i in range(numIterations):
        assignCentroids(kClusters, clusters, attributes, data)
        assignData(clusters, data, attributes)
    return clusters


def assignData(clusters, data, attributes):
    for dataItem in data:
        distances = []
        for currentCluster in clusters:
            distances.append(
                calculateDistance(dataItem, currentCluster['centroid'], attributes)
            )
        clusters[distances.index(min(distances))]['data'].append(dataItem)

def assignCentroids(kClusters, clusters, attributes, data):
    if(not clusters):
        for i in range(kClusters):
            clusters.append({
                'centroid': random.choice(data),
                'data': []
            })
    else:
        for currentCluster in clusters:
            for currentAttribute in attributes:
                if(currentCluster['data']):
                    currentCluster['centroid'][currentAttribute] = reduce(lambda x, y: x + y, map(lambda data: data[currentAttribute], currentCluster['data']))/len(currentCluster['data'])
                    currentCluster['data'] = []
        



def calculateDistance(dataItem1, dataItem2, attributes):
    return np.sqrt(
        reduce(lambda x, y: x + y, map(lambda attribute: (dataItem1[attribute] - dataItem2[attribute]) ** 2, attributes))
    )