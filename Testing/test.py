import numpy as np
from pathlib import Path
from abc import abstractmethod
from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.labelquerier import LabelQuerier
from noise_robust_cobras.metric_learning.metriclearning_algorithms import * 
from metric_learn import *
from pathlib import Path
import os
import pandas as pd
from sklearn.cluster import KMeans
import math
from sklearn.cluster import SpectralClustering
dataset_path = Path('testing/datasets/cobras-paper/UCI/iris.data').absolute()
dataset = np.loadtxt(dataset_path, delimiter=',')
data = dataset[:, 1:]
target = dataset[:, 0]
querier2 = LabelQuerier(None, target, 200)
clusterer = COBRAS(correct_noise=False)
all_clusters, runtimes, superinstances, clusterIteration, *_ = clusterer.fit(data, -1, None, querier2)

querier2 = LabelQuerier(None, target, 200)
clusterer = COBRAS(correct_noise=False, metric = EuclidianDistance())
all_clusters2, *_ = clusterer.fit(data, -1, None, querier2)


import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
x = np.arange(len(all_clusters))
IRA = np.array([adjusted_rand_score(target, np.array(clustering)) for clustering in all_clusters])
IRA_normal = np.array([adjusted_rand_score(target, np.array(clustering)) for clustering in all_clusters2])
x_2 = np.arange(len(all_clusters2))
plt.plot(x, IRA, color='r', label='mCOBRAS')
plt.plot(x_2, IRA_normal, color='g', label='COBRAS')
plt.legend()
plt.show()