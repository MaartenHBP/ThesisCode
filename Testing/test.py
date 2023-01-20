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
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
seeds = [3,25 ,78, 99, 105, 2, 33]
dataset_path = Path('testing/datasets/cobras-paper/UCI/sonar.data').absolute()
dataset = np.loadtxt(dataset_path, delimiter=',')
data = dataset[:, 1:]
target = dataset[:, 0]

mean = np.zeros(200) # hebben eigenlijk andere metrieken nodig om deze dingen te testen (komt wrs op hetzelfde neer)
meanM = np.zeros(200)
meanMeh = np.zeros(200)

for seed in seeds:
    print(seed)
    querier2 = LabelQuerier(None, target, 200)
    clusterer = COBRAS(correct_noise=False, seed=seed, metric = QueriesLearning(when = 'begin', queriesNeeded=32, metric = None, once = True))
    all_clusters3, runtimes, superinstances, clusterIteration, *_ = clusterer.fit(data, -1, None, querier2)
    if len(all_clusters3) < 200:
        diff = 200 - len(all_clusters)
        for ex in range(diff):
            all_clusters3.append(all_clusters3[-1])

    querier2 = LabelQuerier(None, target, 200)
    clusterer = COBRAS(correct_noise=False, seed=seed, metric = QueriesLearning(when = 'begin', queriesNeeded=32, metric = {"value": MMC, "parameters": {}}, once = True))
    all_clusters, runtimes, superinstances, clusterIteration, *_ = clusterer.fit(data, -1, None, querier2)
    if len(all_clusters) < 200:
        diff = 200 - len(all_clusters)
        for ex in range(diff):
            all_clusters.append(all_clusters[-1])

    querier2 = LabelQuerier(None, target, 200)
    clusterer = COBRAS(correct_noise=False, metric = EuclidianDistance(), seed=seed)
    all_clusters2, *_ = clusterer.fit(data, -1, None, querier2)
    if len(all_clusters2) < 200:
        diff = 200 - len(all_clusters2)
        for ex in range(diff):
            all_clusters2.append(all_clusters2[-1])

    meanM += np.array([adjusted_rand_score(target, np.array(clustering)) for clustering in all_clusters])
    mean += np.array([adjusted_rand_score(target, np.array(clustering)) for clustering in all_clusters2])
    meanMeh += np.array([adjusted_rand_score(target, np.array(clustering)) for clustering in all_clusters3])



x = np.arange(200)
plt.plot(x, meanM/len(seeds), color='r', label='mCOBRAS')
plt.plot(x, meanMeh/len(seeds), color='b', label='rebuildOnly')
plt.plot(x, mean/len(seeds), color='g', label='COBRAS')
plt.legend()
plt.show()