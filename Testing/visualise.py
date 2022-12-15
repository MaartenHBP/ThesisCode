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
from matplotlib.animation import FuncAnimation



path = Path(f'testing/datasets/drawn/spectral.data').absolute()
dataset = np.loadtxt(path, delimiter=',')
data = dataset[:, 1:]
target = dataset[:, 0]

querier = LabelQuerier(None, target, 200)

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
clusterer = COBRAS(correct_noise=False, metric_algo=metric_algo, beforeSplitting=True, localLearning=True)
all_clusters, runtimes, *_ = clusterer.fit(data, -1, None, querier)
print(all_clusters)
best_clustering = all_clusters[-1]
print(np.unique(best_clustering))
runtime = runtimes[-1]
# plt.scatter(data[:,0], data[:,1], c = best_clustering)

ARI_score = adjusted_rand_score(target, best_clustering)
print(f"Clustering took {runtime:0.3f}, ARI = {ARI_score:0.3f}")
fig = plt.figure()
def anim(i):
    cluster = i%len(all_clusters)
    fig.clear()
    plt.text(0.15,0.3,str(cluster), fontsize = 22)
    plt.scatter(data[:,0], data[:,1], c = all_clusters[cluster])

animation = FuncAnimation(fig, anim, interval = 500)
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
