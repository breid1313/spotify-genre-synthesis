# import time
# import matplotlib.pyplot as plt
# import numpy as np

# from sklearn.cluster import AgglomerativeClustering
# from sklearn.neighbors import kneighbors_graph

# # Generate sample data
# n_samples = 1500
# np.random.seed(0)
# t = 1.5 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
# x = t * np.cos(t)
# y = t * np.sin(t)


# X = np.concatenate((x, y))
# X += .7 * np.random.randn(2, n_samples)
# X = X.T

# # Create a graph capturing local connectivity. Larger number of neighbors
# # will give more homogeneous clusters to the cost of computation
# # time. A very large number of neighbors gives more evenly distributed
# # cluster sizes, but may not impose the local manifold structure of
# # the data
# knn_graph = kneighbors_graph(X, 30, include_self=False)

# for connectivity in (None, knn_graph):
#     for n_clusters in (30, 3):
#         plt.figure(figsize=(10, 4))
#         for index, linkage in enumerate(('average',
#                                          'complete',
#                                          'ward',
#                                          'single')):
#             plt.subplot(1, 4, index + 1)
#             model = AgglomerativeClustering(linkage=linkage,
#                                             connectivity=connectivity,
#                                             n_clusters=n_clusters)
#             t0 = time.time()
#             model.fit(X)
#             elapsed_time = time.time() - t0
#             plt.scatter(X[:, 0], X[:, 1], c=model.labels_,
#                         cmap=plt.cm.nipy_spectral)
#             plt.title('linkage=%s\n(time %.2fs)' % (linkage, elapsed_time),
#                       fontdict=dict(verticalalignment='top'))
#             plt.axis('equal')
#             plt.axis('off')

#             plt.subplots_adjust(bottom=0, top=.83, wspace=0,
#                                 left=0, right=1)
#             plt.suptitle('n_cluster=%i, connectivity=%r' %
#                          (n_clusters, connectivity is not None), size=17)


# plt.show()


import pandas as pd
from sklearn import datasets
from jqm_cvi.jqmcvi import base
  
# loading the dataset
X = datasets.load_iris()
df = pd.DataFrame(X.data)
  
# K-Means
from sklearn import cluster
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(df) #K-means training
y_pred = k_means.predict(df)
  
# We store the K-means results in a dataframe
pred = pd.DataFrame(y_pred)
pred.columns = ['Type']
  
# we merge this dataframe with df
prediction = pd.concat([df, pred], axis = 1)
  
# We store the clusters
clus0 = prediction.loc[prediction.Species == 0]
clus1 = prediction.loc[prediction.Species == 1]
clus2 = prediction.loc[prediction.Species == 2]
cluster_list = [clus0.values, clus1.values, clus2.values]
  
print(base.dunn(cluster_list))