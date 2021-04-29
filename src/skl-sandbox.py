# -*- coding: utf-8 -*-

"""
Sandbox area to experiment with the clustering methods
available to us in scikit-learn.
Example of the following clustering techniques will follow:

K-Means
Affinity Propagation
Mean-shift
Spectral clustering
Ward hierarchical clustering
Agglomerative clustering
DBSCAN
Gaussian mixtures
Birch
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, Birch, estimate_bandwidth
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from itertools import cycle

# set the location on the data files
DATA_FILE = Path(__file__).parent.parent / "./data/data.csv"

# parent figure
plt.figure(figsize=(12, 12))

# read data into a DataFrame
tracks = pd.read_csv(DATA_FILE)
# use a small subset
tracks = tracks.head(100)

# set X vector data and y to predict
X, y = tracks[["acousticness", "energy"]], tracks["popularity"]

###############
# K Means
###############
y_pred = KMeans(n_clusters=4).fit_predict(X)
# plot
plt.scatter(X["acousticness"], X["energy"], c=y_pred)
plt.title("KMeans")

###############
# Mean Shift
###############
plt.figure(figsize=(12, 12))
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=X.shape[0])
# there has to be a better way to do the below...
# just getting it to work with example plotting syntax
X = np.array([np.array([tracks.loc[n]["acousticness"], tracks.loc[n]["energy"]]) for n in range(tracks.shape[0])])
# X = np.array([tracks["acousticness"], tracks["energy"]])
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(cluster_centers)

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    # syntax below assumes X is numpy.array object
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('MeanShift\nEstimated number of clusters: %d' % n_clusters_)


###############
# Affinity Prop
###############
plt.figure(figsize=(12, 12))
# time and space complexity on this one are brutal
# maxed out my machine with the full dataset

X = np.array([np.array([tracks.loc[n]["acousticness"], tracks.loc[n]["energy"]]) for n in range(tracks.shape[0])])
y_pred = AffinityPropagation().fit(X)
cluster_center_indices = y_pred.cluster_centers_indices_
labels = y_pred.labels_
n_clusters = len(cluster_center_indices)
# plot
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters), colors):
    class_members = labels == k
    cluster_center = X[cluster_center_indices[k]]
    test = X[class_members, 0]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Affinity Prop\nEstimated number of clusters: %d' % n_clusters_)


###############
# Agglomeritive
###############
plt.figure(figsize=(12, 12))
X = np.array([np.array([tracks.loc[n]["acousticness"], tracks.loc[n]["energy"]]) for n in range(tracks.shape[0])])
n_samples, n_features = X.shape

for index, linkage in enumerate(('ward', 'average', 'complete')):
    plt.subplot(221 + index)
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=2)
    clustering.fit(X)
    labels = clustering.labels_
    plt.title("Agglomeritive, %s linkage" % linkage)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.nipy_spectral)


###############
# DBSCAN
###############

# not quite going yet

plt.figure(figsize=(12, 12))
X = np.array([np.array([tracks.loc[n]["acousticness"], tracks.loc[n]["energy"]]) for n in range(tracks.shape[0])])

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
plt.title("DBSCAN, %s clusters" % n_clusters)
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.show()