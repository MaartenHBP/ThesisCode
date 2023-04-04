from array import array
from audioop import mul
from math import sqrt
import math
import os
import functools
from pathlib import Path
import json
from datetime import datetime

from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.noisy_labelquerier import ProbabilisticNoisyQuerier
from noise_robust_cobras.querier.labelquerier import LabelQuerier
from noise_robust_cobras.metric_learning.module.metriclearners import *
from noise_robust_cobras.metric_learning.module.metriclearning_algorithms import *
from noise_robust_cobras.clustering_algorithms.clustering_algorithms import *
from noise_robust_cobras.strategies.splitlevel_estimation import *
import copy
from sklearn.model_selection import StratifiedKFold, KFold

import numpy as np 
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
import sklearn as sk
from statistics import mean
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

from dask.distributed import Client, LocalCluster

import shutil

from sklearn.manifold import TSNE
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import ConvexHull

nbRUNS = 100
ARGUMENTS = range(100)
SEED = 24
random_generator = np.random.default_rng(SEED)
seeds = [random_generator.integers(1,1000000) for i in range(nbRUNS)] # creation of the seeds

ABSOLUTE = 200
RELATIVE = 0.3

PATH_COBRAS = "experimenten/COBRAS"

def createFolds(labels, index:int):
    fold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seeds[index])

    training_indices = []
    testing_indices = []
    
    for fold_nb, (train_indices, test_indices) in enumerate(fold.split(np.zeros(len(labels)), labels)):
        training_indices.append(train_indices.tolist())
        testing_indices.append(train_indices.tolist())

#################
# Parallel code #
#################

def runCOBRAS(dataName, seed, arguments):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    path = Path(f'datasets/cobras-paper/UCI/{dataName}.data').absolute()
    dataset = np.loadtxt(path, delimiter=',')
    data = dataset[:, 1:]
    target = dataset[:, 0]

    querylimit = max(math.floor(len(data)*RELATIVE), ABSOLUTE)
    runlimit = min(querylimit, len(data))


    querier = LabelQuerier(None, target, runlimit)
    clusterer = COBRAS(correct_noise=False, seed=seeds[seed], **arguments)

    all_clusters, _, _, _, _, _, _ = clusterer.fit(data, -1, None, querier)

    if len(all_clusters) < querylimit:
        diff = querylimit - len(all_clusters)
        for ex in range(diff): all_clusters.append(all_clusters[-1])

    return [adjusted_rand_score(target, np.array(clustering)) for clustering in all_clusters]


def makeARI(path, name_algo = ""): # momenteel enkel vergelijken met COBRAS, en ook nog enkel absolute: TODO
    test = loadDict(path, "total")
    cobras = loadDict(PATH_COBRAS, "total")
    
    cobraspd = pd.DataFrame()
    testpd = pd.DataFrame()

    if not name_algo:
        name_algo = path


    for key, item in testpd.items():

        testpd[key] = np.array(item)[:200]
        cobraspd[key] = np.array(cobras[key])[:200]
        
        plt.plot(cobraspd[key], label = "COBRAS")
        plt.plot(testpd[key], label = name_algo)

        plt.title(key)
        plt.xlabel("#queries")
        plt.ylabel("ARI")
        plt.ylim((0,1))
        plt.legend()
        plt.savefig(f"{path}/plots/{key}.png")
        plt.clf()

    all_results = pd.DataFrame()
    all_results["COBRAS"] = cobraspd.mean(axis=1)
    all_results[name_algo] = testpd.mean(axis=1)

    all_results.plot(xlabel="#queries", ylabel="ARI", ylim = (0,1))
    plt.savefig(f"{path}/plots/total.png")

def makeDifferencePlot(path, name_algo = "")



# loadData
def loadDict(path, name):
    path_test = Path(f'{path}/{name}.json').absolute()
    if not os.path.exists(path_test):
        return dict()
    print(f'{path}/{name}.json')
    with open(f'{path}/{name}.json') as json_file:
        # ga verder met een experiment of start
        return json.load(json_file)
