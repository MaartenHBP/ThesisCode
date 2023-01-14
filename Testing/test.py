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