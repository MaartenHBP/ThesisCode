from pathlib import Path
import time
import numpy as np
from metric_learn import *
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import plotly_express as px
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.noisy_labelquerier import ProbabilisticNoisyQuerier
from noise_robust_cobras.querier.labelquerier import LabelQuerier
from noise_robust_cobras.metric_learning.metriclearning_algorithms import SemiSupervisedMetric
from noise_robust_cobras.metric_learning.metriclearning_algorithms import SupervisedMetric
from noise_robust_cobras.metric_learning.metricLearners import *
from matplotlib.animation import FuncAnimation
from scipy import interpolate
from scipy.spatial import ConvexHull
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import SpectralClustering


plt.style.use("default")
path = Path(f'testing/datasets/drawn/spectral.data').absolute()
# path = Path(f'testing/datasets/drawn/simple.data').absolute()
dataset = np.loadtxt(path, delimiter=',')
data = dataset[:, 1:]
target = dataset[:, 0]

# data = gb_lmnn_class().fit(data,target).transform(data)

# cov = Covariance().fit(data)
# x = cov.transform(data)

# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection='3d')

# embed = gb_lmnn(data, target, k =  3, L = None, n_trees=200, verbose=True, xval =None, yval = None)
# newdata = embed.transform(np.copy(data))

# import numpy as np
# from metric_learn import LMNN

# lmnn = ITML_Supervised()
# lmnn.fit(data, target)
# data = lmnn.transform(data)
# ax.scatter(data[:,0], data[:,1], data[:,2], c = target)
# plt.show()



# querier = LabelQuerier(None, target, 200)
# clusterer = COBRAS(correct_noise=False)
# all_clusters, runtimes, superinstances, clusterIteration, transformations, ml, cl = clusterer.fit(data, -1, None, querier)

# le = LE(data, dim = 2, k = 3, graph = 'k-nearest', weights = 'heat kernel', 
#         sigma = 5, laplacian = 'symmetrized')
# Y = le.transform()

# le.plot_embedding_2d(colors = target)

# print(le._W)

# querier = LabelQuerier(None, target, 200)
# clusterer = COBRAS(correct_noise=False)
# all_clusters, runtimes, superinstances, clusterIteration, transformations, ml, cl = clusterer.fit(data, -1, None, querier)

# embedding = SpectralEmbedding(n_components=2)
# X_transformed = embedding.fit_transform(data)
# plt.scatter(Y[:,0], Y[:,1], c = target)
# plt.show()

# from sklearn.neighbors import radius_neighbors_graph
# W = radius_neighbors_graph(data,0.4,mode='connectivity', metric='minkowski', p=2, metric_params=None, include_self=False).todense()

# print(W)

# # max = W.max()
# # vectorizer = np.vectorize(lambda x: 1 if x/max < 0.2 else 0)
# # W = np.vectorize(vectorizer)(W)

# W_reserve = np.copy(W)

# for i in range(len(ml)):
#         W[ml[i][0],ml[i][1]] = 1
#         W[ml[i][1],ml[i][0]] = 1
# for i in range(len(cl)):
#         W[cl[i][0],cl[i][1]] = 0
#         W[cl[i][1],cl[i][0]] = 0


# # degree matrix
# from scipy.sparse import csgraph
# L = csgraph.laplacian(W, normed=False)
# L_without = csgraph.laplacian(W_reserve, normed=False)

# print('laplacian matrix:')
# print(L)

# e, v = np.linalg.eig(L)
# print(v.shape)
# print(len(target))
# ew, vw = np.linalg.eig(L_without)

# plt.scatter(v[:,0], v[:,1], c = target)
# plt.show()
# plt.scatter(vw[:,0], vw[:,1], c = target)
# plt.show()
# # embed = gb_lmnn(data, target, k =  3, L = None, n_trees=200, verbose=True, xval =None, yval = None)

# embedding = SpectralEmbedding(n_components=2)
# X_transformed = embedding.fit_transform(data)
# plt.scatter(X_transformed[:,0], X_transformed[:,1], c = target)
# plt.show()
# newData = data# embed.transform(data)

# clustering = SpectralClustering(n_clusters=2,
# assign_labels='discretize',
# random_state=0).fit(data).labels_
# # mmc = LMNN()
# # mmc.fit(data, target)
# # newData = mmc.transform(np.copy(data))
# X_transformed = embedding.fit_transform(data)
# plt.scatter(X_transformed[:,0], X_transformed[:,1], c = target)
# plt.show()
# embedding = SpectralEmbedding(n_components=2)
# X_transformed = embedding.fit_transform(X_transformed)
# plt.scatter(X_transformed[:,0], X_transformed[:,1], c = target)
# plt.show()
# embedding = SpectralEmbedding(n_components=2)
# X_transformed = embedding.fit_transform(X_transformed)
# plt.scatter(X_transformed[:,0], X_transformed[:,1], c = target)
# plt.show()
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X_transformed).labels_
# plt.scatter(X_transformed[:,0], X_transformed[:,1], c = target)
# plt.show()
# plt.scatter(data[:,0], data[:,1], c = kmeans)
# plt.show()
# plt.scatter(data[:,0], data[:,1], c = clustering)
# plt.show()
# plt.scatter(data[:,0], data[:,1], c = kmeans)
# plt.show()
# plt.scatter(newData[:,0], newData[:,1], c = target)


# plt.show()

querier = LabelQuerier(None, target, 20)

metric_algo = {
                "type": "class",
                "value": SemiSupervisedMetric,
                "parameters": {
                    "algo": {
                        "value": ITML,
                        "parameters": {
                            "gamma": 1.0,
                            "max_iter": 100,
                            "convergence_threshold": 0.001,
                            "prior": "identity",
                            "verbose": False,
                            "random_state": 42
                        }
                    },
                    "steps": 0
                }
            }

# make new COBRAS
clusterer2 = COBRAS(correct_noise=False, end = True, metric_algo=metric_algo)
all_clusters, runtimes, superinstances, clusterIteration, *_ = clusterer2.fit(data, -1, None, querier)
plt.scatter(clusterer2.data[:,0], clusterer2.data[:,1], c = target)
plt.show()
querier2 = LabelQuerier(None, target, 200)
clusterer = COBRAS(correct_noise=False, end = True, metric_algo=metric_algo, logExtraInfo=True)
all_clusters, runtimes, superinstances, clusterIteration, *_ = clusterer.fit(clusterer2.data, -1, None, querier2)
best_clustering = all_clusters[-1]
runtime = runtimes[-1]
# plt.scatter(data[:,0], data[:,1], c = best_clustering)

ARI_score = adjusted_rand_score(target, best_clustering)
print(f"Clustering took {runtime:0.3f}, ARI = {ARI_score:0.3f}")
fig = plt.figure()
def anim(i):
    cluster = i%len(superinstances)
    fig.clear()
    plt.text(0.15,0.3,str(cluster), fontsize = 22)
    plt.scatter(data[:,0], data[:,1], c = clusterIteration[cluster])

    for j in np.unique(superinstances[cluster]):
    # get the convex hull
        points = data[superinstances[cluster] == j]
        if len(points) < 3:
            continue
        hull = ConvexHull(points)
        x_hull = np.append(points[hull.vertices,0],
                        points[hull.vertices,0][0])
        y_hull = np.append(points[hull.vertices,1],
                        points[hull.vertices,1][0])
        
        # interpolate
        # dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
        # dist_along = np.concatenate(([0], dist.cumsum()))
        # spline, u = interpolate.splprep([x_hull, y_hull], 
        #                                 u=dist_along, s=0, per=1)
        # interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
        # interp_x, interp_y = interpolate.splev(interp_d, spline)
        # plot shape
        plt.fill(x_hull, y_hull, '--', alpha=0.2)

animation = FuncAnimation(fig, anim, interval = 1000)
plt.show()
# dataset = np.loadtxt(path, delimiter=',')
# data = dataset[:, 1:]
# target = dataset[:, 0]
# # plt.scatter(data[:,0], data[:,1])
# # plt.show()
# d = {'iteration': [], 'x': [], 'y': [], 'point' : []}
# transformed = pd.DataFrame(data = d)
# fig, ax = plt.subplots()
# kmeans = KMeans(n_clusters=2, random_state=0).fit(data).labels_
# print(kmeans)
# ax.scatter(data[:,0][kmeans == 0], data[:,1][kmeans == 0], color = "red")
# ax.scatter(data[:,0][kmeans == 1], data[:,1][kmeans == 1], color = "blue")

# st.pyplot(fig)
# # for i in [2,3,4,5,6,7,8,9]:
# mmc = NCA()
# mmc.fit(data, target)
# newData = mmc.transform(np.copy(data))
# fig, ax = plt.subplots()
# kmeans = KMeans(n_clusters=2, random_state=0).fit(newData).labels_
# print(kmeans)
# ax.scatter(data[:,0][kmeans == 0], data[:,1][kmeans == 0], color = "red")
# ax.scatter(data[:,0][kmeans == 1], data[:,1][kmeans == 1], color = "blue")

# st.pyplot(fig)
# new_row = pd.DataFrame({'iteration':i, 'x':np.copy(data[:,0]), 'y':np.copy(data[:,1]), 'point':np.arange(len(target))})
# transformed = transformed.append(new_row)
# print(transformed)
# fig = px.scatter(transformed, x = "x", y = "y", animation_frame="iteration", animation_group='point')
# st.write(fig)
