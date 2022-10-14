import os
import numpy as np
from metric_learn import ITML
from metric_learn import MMC
from metric_learn import RCA
from metric_learn import SDML
from metric_learn import LMNN
from metric_learn import NCA
from pathlib import Path
from sklearn.metrics import adjusted_rand_score
from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.noisy_labelquerier import ProbabilisticNoisyQuerier
from noise_robust_cobras.querier.labelquerier import LabelQuerier
from noise_robust_cobras.metric_learning.metriclearning_algorithms import SemiSupervisedMetric
from noise_robust_cobras.metric_learning.metriclearning_algorithms import SupervisedMetric
path = Path('datasets/cobras-paper/').absolute()
dir_list = os.listdir(path)
string = ""
averageMetric = 0
averageNormal = 0
n = 0
for i in dir_list:
    p = os.path.join(path, i)
    dataset = np.loadtxt(p, delimiter=',')
    data = dataset[:, 1:]
    target = dataset[:, 0]

    querier = LabelQuerier(None, target, 100)

    # make new COBRAS
    clusterer = COBRAS(correct_noise=False , metric_algo=SemiSupervisedMetric(), end = True) # baseline
    all_clusters, runtimes, *_ = clusterer.fit(data, -1, None, querier)
    newData = np.copy(clusterer.data)

    averageNormal += adjusted_rand_score(target, best_clustering)

    # make new COBRAS
    querier = LabelQuerier(None, target, 100)
    clusterer = COBRAS(correct_noise=False , metric_algo=SemiSupervisedMetric())
    all_clusters, runtimes, *_ = clusterer.fit(newData, -1, None, querier)
    
    best_clustering = all_clusters[-1]
    runtime = runtimes[-1]

    averageNormal += adjusted_rand_score(target, best_clustering)
    n += 1

averageMetric /= n
averageNormal /= n

print("normal: " + str(averageNormal))
print("metric: " + str(averageMetric))