# -*- coding: utf-8 -*-

import os
import pandas as pd 
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict
from math import inf, sqrt
from copy import deepcopy

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# credit for the Dunn's index measure:
# source belongs to Joaquim Viegas and is used under The MIT License 
# https://github.com/jqmviegas/jqm_cvi
from jqm.jqmcvi import base

import matplotlib.pyplot as plt

# set the location on the data files
DATA_FILE = Path(__file__).parent.parent / "./data/data.csv"
DATA_BY_GENRE = Path(__file__).parent.parent / "./data/data_by_genres.csv"
DATA_BY_ARTIST = Path(__file__).parent.parent / "./data/data_by_artist.csv"
DATA_BY_YEAR = Path(__file__).parent.parent / "./data/data_by_year.csv"
DATA_WITH_GENRE = Path(__file__).parent.parent / "./data/data_w_genres.csv"

"""
Start exploring the data.
With DataFrame.head() we get a glimpse into the first few rows.
With DataFrame.info() we can check for missing data types
and see the datatypes that read_csv() cast each column to.
"""

# read tracks
tracks = pd.read_csv(DATA_FILE)
print("Track data:")
print(tracks.head())
print(tracks.info())
print(tracks.columns)
print(tracks.describe())

# dev, truncate the data to 20k
tracks = tracks.head(20000)
# copy for later access (at end)
tracks_orig = deepcopy(tracks)

"""
Check for null values. If we detect any, we should consider providing
an estimated value or discarding the row from the analysis.
"""

nulls = tracks.isnull()
manual_check = []
for row in nulls.itertuples():
    if (True in row[1:]):
        print("Null value detected at the following row: {0}.\n"
        "Manual review is required.".format(row))
        manual_check.append(row)
if not manual_check:
    print("No null values are present in the data.")
# delete nulls to save space. we're done with it
del nulls


""" 
Check for outliers. If a row has any data that is a significant outlier, we should exclude 
that row from our analysis. A serious outlier will affect the results of the analysis
and prohibit a useful normalization of the data
"""
print("Running outlier detection. For each column, rows that fall outside 3 std deviation will be discarded.\n"
"A row will be discarded if it violates this condition for any column.")
old_length = tracks.shape[0]
no_normalize = ["artists", "explicit", "id", "key", "mode", "name", \
    "release_date", "year"]
for column in tracks.columns:
    if column not in no_normalize:
        tracks[((tracks[column] - tracks[column].mean()) / tracks[column].std()).abs() < 3]
        
new_length = tracks.shape[0]
print("Removed {0} rows containing outliers".format(old_length - new_length))

"""
Normalize the data. For most attributes, we should work on a consisten scale of [0,1]
so our analysis does not yield additional weight to attributes like tempo
or loudness that have higher numerical values.
"""
scalar = MinMaxScaler() # defaulted to [0,1]

# some columns dont make sense to normalize
no_normalize = ["artists", "explicit", "id", "key", "mode", "name", \
    "release_date"]
for column in tracks.columns:
    if column not in no_normalize:
        tracks[column] = scalar.fit_transform(np.array(tracks[column]).reshape(-1,1))
        # verify that the changes were successful
        if max(tracks[column]) > 1 or min(tracks[column]) < 0:
            print("Failed to normalize {0} column!".format(column))
print("Skipped data normalization for these columns: {0}".format(no_normalize))
print("Data normalized successfully!")
print(tracks.describe(include='all'))


"""
Run Principle Component Analysis and agglomerative clustering. This casts the data down into a lower dimensional space.
We wish to do this for ease of computation and visualization.
"""

scores = defaultdict(defaultdict)
dunn_scores = defaultdict(defaultdict)
N_COMPONENTS = 2
try:
    tracks = tracks.drop(columns=no_normalize)
except KeyError:
    pass

# get a PCA helper
# start simple with two components
pca = PCA(n_components=N_COMPONENTS)

# extract the values from the dataframe
X = tracks.values
# get the principle components
components = pca.fit_transform(X)
# load into a new dataframe
principle_components = pd.DataFrame(data=components, columns=["First Principle Component", "Second Principle Component"])

# save the points
points = []
for index, row in principle_components.iterrows():
    points.append([row["First Principle Component"], row["Second Principle Component"]])
points = np.array(points)

print("Data has been reduced into {0} components.".format(N_COMPONENTS))
print(principle_components.head(10))


# save scores so we dont have to loop every time to get it 
# scores = {2: 0.29779922276003384, 3: 0.2895338144661002, 4: 0.1986653470600852, 5: 0.20496256521574424, 6: 0.14288880165471826, 7: 0.11947392560701121, 8: 0.090110379941736, 9: 0.09075607364077816, 10: 0.08394602352603533, 11: 0.06974414140033265, 12: 0.06211648810417688, 13: 0.06597052041038544, 14: 0.0617226032663908, 15: 0.054445644743169194, 16: 0.04529505675603273, 17: 0.04307463286807813, 18: 0.04391162094083206, 19: 0.03769033082220651, 20: 0.02537174450734863, 21: 0.025896713480092683, 22: 0.02618329067337586, 23: 0.014433151116067995, 24: 0.01477813745818307, 25: 0.011612529458209927, 26: 0.011501369852073489, 27: 0.014018531853167999, 28: 0.014372122355856283, 29: 0.013608603637375143, 30: 0.010252333404633333, 31: 0.011398193853021711, 32: 0.010697359456563148, 33: 0.003673324753481274, 34: 0.002661612146320156, 35: 0.0031635987466547165, 36: 0.0015658094096856424, 37: 0.0019805526398985964, 38: -0.0005452559162407479, 39: -0.00022816883119231832, 40: -0.005157732274158492, 41: -0.00846875693340036, 42: -0.0121249356291998, 43: -0.013995545349073808, 44: -0.01418401171404668, 45: -0.015046041779517216, 46: -0.015336880326188018, 47: -0.0166517110652135, 48: -0.016322294721026934, 49: -0.017459815895635448, 50: -0.017130968250830124, 51: -0.022605121071722, 52: -0.022406544602353678, 53: -0.024681186463570173, 54: -0.024757654848462683, 55: -0.02705737263735006, 56: -0.026917847733628618, 57: -0.026682228817120297, 58: -0.02780662410209675, 59: -0.02760515202533527, 60: -0.027762310739872863, 61: -0.02755525687308282, 62: -0.028505640763263874, 63: -0.02898890731978709, 64: -0.028774561333129042, 65: -0.028671706074539588, 66: -0.028741432575369725, 67: -0.02948222737065004, 68: -0.030782021933386187, 69: -0.03162373596044226, 70: -0.031517029896421224, 71: -0.032993900872837184, 72: -0.03413682006315136, 73: -0.03661442726316393, 74: -0.03630972413715104, 75: -0.03668144334348229, 76: -0.03801115187765717, 77: -0.041743950760992424, 78: -0.04157013739873743, 79: -0.04156515716587346, 80: -0.04146785997880129, 81: -0.041344172636060765, 82: -0.04263340075718238, 83: -0.04272596112741224, 84: -0.04253048744702642, 85: -0.042853283691735626, 86: -0.045045914560491715, 87: -0.04503785777832566, 88: -0.04591402913524653, 89: -0.04586063455042016, 90: -0.04558969836363881, 91: -0.0454656111196773, 92: -0.04602098155225899, 93: -0.04691752158055329, 94: -0.04679740504881781, 95: -0.04700810994015831, 96: -0.047063618418042394, 97: -0.047061439204236485, 98: -0.0474347741861287, 99: -0.04735169750092784}
print(sorted(scores))

# adding each cluster reduces the score because the data is so dense
# we should choose a number that is a reasonable number of genres.
# choose 11
print("Main analysis. Setting clusters to 11...")
N_CLUSTERS = 11

clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS)
y_clustering = clustering.fit_predict(components)
labels = clustering.labels_

dunn = base.dunn_fast(points, y_clustering)
print("Dunn: {}".format(dunn))
dunn_scores[N_CLUSTERS] = dunn

sil = silhouette_score(X, labels, metric="euclidean")
scores[N_CLUSTERS] = sil

print("For PCA2 and n_clusters=11:")
print("Dunn = {}".format(dunn))
print("Silhouette = {}".format(sil))


plt.figure(figsize=(8,8))
plt.bar(range(len(scores)), list(scores.values()), align="center")
ax = plt.axes()
ax.set_title("Silhouette Score vs Number of Clusters")

plt.figure(figsize=(8,8))
plt.bar(range(len(dunn_scores)), list(dunn_scores.values()), align="center")
ax = plt.axes()
ax.set_title("Dunn Index vs Number of Clusters")

# uncomment below later on to get graphs and stuff

N_COMPONENTS = 2
print("Decomposing the data into 2 principle components")
# drop some of the columns that we don't care about
try:
    tracks = tracks.drop(columns=no_normalize)
except KeyError:
    pass




# get a PCA helper
# start simple with two components
pca = PCA(n_components=N_COMPONENTS)

#########
# getting PCA values
#########
# N_COMPONENTS = tracks.shape[1]
pca = PCA(n_components=N_COMPONENTS)
X = tracks.values
components = pca.fit_transform(X)

fig = plt.figure(figsize=(10,10))
ax = plt.axes()
ax.set_title("PCA - Variance by Component")
ax.set_xlabel("Principle Components")
ax.set_ylabel("Explained Variance Ratio")
ratio = pca.explained_variance_ratio_
plt.bar([i for i in range(1, len(pca.explained_variance_ratio_)+1)], list(pca.explained_variance_ratio_))

# extract the values from the dataframe
X = tracks.values
# get the principle components
components = pca.fit_transform(X)
# load into a new dataframe
principle_components = pd.DataFrame(data=components, columns=["First Principle Component", "Second Principle Component"])

# save the points
points = []
for index, row in principle_components.iterrows():
    points.append([row["First Principle Component"], row["Second Principle Component"]])
points = np.array(points)


print("Data has been reduced into {0} components.".format(N_COMPONENTS))
print(principle_components.head(10))


"""
Perform the agglomerative clustering.
"""
print("Clustering the data...")

clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS)
y_clustering = clustering.fit_predict(components)
labels = clustering.labels_

scores["PCA2"] = silhouette_score(X, labels, metric="euclidean")

pca2_results = defaultdict()

pca2_df = pd.concat([principle_components, pd.DataFrame(y_clustering, columns=["Label"])], axis=1)
# for each cluster
for i in range(N_CLUSTERS):
    cluster = pca2_df.loc[pca2_df["Label"]==i]
    # compute the x and y average
    x_avg = cluster["First Principle Component"].sum() / cluster.shape[0]
    y_avg = cluster["Second Principle Component"].sum() / cluster.shape[0]
    # brute force find the closest point
    min_dist = inf
    center = None
    for point in points:
        dist = sqrt((x_avg - point[0])**2 + (y_avg - point[1])**2)
        if dist <= min_dist:
            min_dist = dist
            center = point
    # get the index of the center point
    # center = pca2_df.loc[pca2_df["First Principle Component"]==center[0], pca2_df["Second Principle Component"]==center[1]]
    center = pca2_df[(pca2_df["First Principle Component"]==center[0]) & (pca2_df["Second Principle Component"]==center[1])]
    index = center.index.values
    # if two points are the same, we just take one
    index = index[0]
    pca2_results["cluster_"+str(i)] = tracks_orig.iloc[[index]]
for key, value in pca2_results.items():
    print(str(key) + ": " + str(value))

    

fig = plt.figure(figsize=(10,10))
ax = plt.axes()
ax.set_title("PCA - n=2")
ax.set_xlabel("First Principle Component")
ax.set_ylabel("Second Principle Component")
plt.scatter(principle_components["First Principle Component"], principle_components["Second Principle Component"], c=labels, cmap=plt.cm.nipy_spectral)


"""
Run Principle Component Analysis and agglomerative clustering. This casts the data down into a lower dimensional space.
We wish to do this for ease of computation and visualization.
"""
N_COMPONENTS = 3
print("Decomposing the data into 3 principle components")


# get a PCA helper
# start simple with two components
pca = PCA(n_components=N_COMPONENTS)

# extract the values from the dataframe
X = tracks.values
# get the principle components
components = pca.fit_transform(X)
# load into a new dataframe
principle_components = pd.DataFrame(data=components, columns=["First Principle Component", "Second Principle Component", "Third Principle Component"])

# save the points
points = []
for index, row in principle_components.iterrows():
    points.append([row["First Principle Component"], row["Second Principle Component"], row["Third Principle Component"]])
points = np.array(points)

print("Data has been reduced into {0} components.".format(N_COMPONENTS))
print(principle_components.head(10))


"""
Perform the agglomerative clustering.
"""
print("Clustering the data...")

clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS)
y_clustering = clustering.fit_predict(components)
labels = clustering.labels_

dunn = base.dunn_fast(points, y_clustering)

sil = silhouette_score(X, labels, metric="euclidean")

print("For PCA3 and n_clusters=11:")
print("Dunn = {}".format(dunn))
print("Silhouette = {}".format(sil))

fig = plt.figure(figsize=(10,10))
ax = plt.axes(projection="3d")
ax.set_title("PCA - n=3")
ax.set_xlabel("First Principle Component")
ax.set_ylabel("Second Principle Component")
ax.set_zlabel("Third Principle Component")
plt.scatter(principle_components["First Principle Component"], principle_components["Second Principle Component"], \
    principle_components["Third Principle Component"], c=labels, cmap=plt.cm.nipy_spectral)



# """
# Run Agglomerative clustering
# """
# print("clustering the data without PCA...")
# X = tracks.values

# clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS)
# y_clustering = clustering.fit_predict(X)
# labels = clustering.labels_

# sil = silhouette_score(X, labels, metric="euclidean")

# dunn = base.dunn_fast(points, y_clustering)
# print("Dunn: {}".format(dunn))

# print("For PCA3 and n_clusters=11:")
# print("Dunn = {}".format(dunn))
# print("Silhouette = {}".format(sil))

# print all scores
print(scores)

#### display the plot
plt.show()
