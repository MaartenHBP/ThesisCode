from array import array
from audioop import mul
from math import sqrt
import math
import os
import functools
from pathlib import Path
import json
from datetime import datetime

from noise_robust_cobras.rebuild_algorithms.rebuild_algorithms import (
    SemiCluster,
    ClosestRebuild,
    Rebuilder,
    ClosestVote
)

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

absolute_testing = np.arange(20, 200, step = 20).tolist()
relative_testing = np.arange(0.1, 1, 0.1).tolist()

PATH_COBRAS = "experimenten/COBRAS"

def createFolds(labels, index:int): # gaan eerst zonder folds testen, een goeie methode gaat deze test ook krijgen
    fold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seeds[index])

    training_indices = []
    testing_indices = []
    
    for fold_nb, (train_indices, test_indices) in enumerate(fold.split(np.zeros(len(labels)), labels)):
        training_indices.append(train_indices.tolist())
        testing_indices.append(test_indices.tolist())

#################
# Parallel code #
#################

def runCOBRAS(seed, dataName, arguments):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    path = Path(f'datasets/cobras-paper/UCI/{dataName}.data').absolute()
    dataset = np.loadtxt(path, delimiter=',')
    data = dataset[:, 1:]
    target = dataset[:, 0]

    # querylimit = max(math.floor(len(data)*RELATIVE), ABSOLUTE)
    querylimit = 500
    # runlimit = min(querylimit, len(data))
    runlimit = querylimit


    querier = LabelQuerier(None, target, runlimit)
    # splitlevel_strategy = ConstantSplitLevelEstimationStrategy(4), doen dit gehardcoded
    clusterer = COBRAS(splitlevel_strategy = ConstantSplitLevelEstimationStrategy(4), correct_noise=False, seed=seeds[seed], **arguments)

    all_clusters, _, _, _, = clusterer.fit(data, -1, None, querier)

    if len(all_clusters) < querylimit:
        diff = querylimit - len(all_clusters)
        for ex in range(diff): all_clusters.append(all_clusters[-1])

    return [adjusted_rand_score(target, np.array(clustering)) for clustering in all_clusters]

###############
# Experiments #
###############
                
def test():
    dataset = "sonar"
    # args = {
    #     "rebuildPhase": True, 
    #     "rebuildLevel": "all", 
    #     "rebuildAmountQueriesAsked" : 150,
    #     "rebuildMetric": False,
    #     "rebuilder": ClosestVote}

    # args = {
    # "metricAmountQueriesAsked": 100,
    # "learnAMetric": True
    # }

    args = {
        "useNewConstraintIndex": True
    }
    # plt.plot(runCOBRAS(55, dataset, {"keepSupervised":True, "rebuildPhase": True, "rebuildLevel": "superinstance", "rebuilder" : SemiCluster,
    #     "rebuildAmountQueriesAsked" : 100, "rebuildMetric":True, "rebuildSuperInstanceLevel": 3}), label = "test_metric")
    # plt.plot(runCOBRAS(55, dataset, {
    #     "keepSupervised":True, 
    #     "rebuildPhase": True, 
    #     "rebuildLevel": "superinstace", 
    #     "rebuilder" : SemiCluster,
    #     "rebuildAmountQueriesAsked" : 75, 
    #     "rebuildMetric":False, 
    #     "rebuildSuperInstanceLevel": 3,
    #     "rebuildPartition": True,
    #     "rebuildPartitionDecider": "vote"}), label = "test")
    # plt.plot(runCOBRAS(55, dataset, {"keepSupervised":True}), label = "COBRAS")

    # plt.legend()
    

    # plt.show()    
    plt.plot(runCOBRAS(20, dataset, args), label = "test")
    print("next")
    # args["rebuildMetric"] = True
    # plt.plot(runCOBRAS(16, dataset, args), label = "test_metric")
    plt.plot(runCOBRAS(20, dataset, {"keepSupervised":True}), label = "COBRASLabels")
    # print("next")
    # plt.plot(runCOBRAS(16, dataset, {}), label = "COBRAS")

    plt.legend()
    

    plt.show()
    
    
######################
# SImple_experiments #
######################
def normalCOBRAS():
    path = Path(f"experimenten/thesis/Chapter5/variance_analysis/splitlevel4").absolute()
    run({ "useNewConstraintIndex" : True,
        "mergeBlobs" : False}, path)


def run(args, path):
    CHECK_FOLDER = os.path.isdir(path)
    if not CHECK_FOLDER:
        os.makedirs(path)
        print("created folder : ", path)
    
    if "rebuilder" in args:
        value = args["rebuilder"]
        args["rebuilder"] = str(args["rebuilder"])
        saveDict(args, path, "settings")
        args["rebuilder"] = value
    else:
        saveDict(args, path, "settings")

    try:
        with LocalCluster() as cluster, Client(cluster) as client:
            path_datasets = Path('datasets/cobras-paper/UCI').absolute()
            datasets = os.listdir(path_datasets)
            run = dict()
            variance = dict()
            p = Path(f'{path}/total.json').absolute()
            if os.path.exists(p):
                run = loadDict(path, f"total")
                variance = loadDict(path, f"variance")
            # saveDict(cobras, f"experimenten/presentatie3", "NORMAL_LMNN")
            for j in range(len(datasets)):
                nameData = datasets[j][:len(datasets[j]) - 5]
                if nameData in run:
                    continue
                print(f"({path})\t ({nameData})\t Running")
                parallel_func = functools.partial(runCOBRAS, dataName = nameData, arguments = args)
                futures = client.map(parallel_func, ARGUMENTS)
                # parallel_func(16)
                results = np.array(client.gather(futures))
                run[nameData] = np.mean(results, axis=0).tolist()
                variance[nameData] = np.std(results, axis=0).tolist()
                saveDict(run, path, "total")
                saveDict(variance, path, "variance")
            saveDict(run, path, "total")
    except Exception as x:
        print("error cccured:" + path)
        errordict = {"problem": str(x)}
        saveDict(errordict, path, "error")




###############
# Plot makers #
###############

def makeARI(path, name_algo = ""): # momenteel enkel vergelijken met COBRAS, en ook nog enkel absolute: TODO
    test = loadDict(path, "total")
    cobras = loadDict(PATH_COBRAS, "total")
    
    cobraspd = pd.DataFrame()
    testpd = pd.DataFrame()

    if not name_algo:
        name_algo = path


    for key, item in test.items():

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

    all_results.plot(xlabel="#queries", ylabel="ARI", ylim = (0.4,0.85))
    # plt.show()
    plt.savefig(f"{path}/plots/total.png")

    plt.clf()

def makeDifferencePlot(path, name_algo = ""):
    test = loadDict(path, "total")
    cobras = loadDict(PATH_COBRAS, "total")
    
    total = np.zeros(200)

    if not name_algo:
        name_algo = path


    for key, item in test.items():

        test_item = np.array(item)[:200]
        cobras_item = np.array(cobras[key])[:200]
        
        bools = test_item > cobras_item

        total += bools

    
    plt.plot(total, label = name_algo)

    plt.title("Better per dataset")
    plt.xlabel("#queries")
    plt.ylabel("#better")
    plt.ylim((0, 16))
    plt.legend()
    plt.savefig(f"{path}/plots/better.png")

    plt.clf()

def rank(paths, names, location, useVariance = False):
    # cobras = loadDict(PATH_COBRAS, "total")

    mean = pd.DataFrame()

    variance = pd.DataFrame()

    ARI = pd.DataFrame()


    # for key, item in cobras.items():
    #     mean[key] = np.array(item)
    
    for i in range(len(paths)):
        path = paths[i]
        testpd = pd.DataFrame()
        variancepd = pd.DataFrame()
        test = loadDict(path, "total")
        var = loadDict(path, "variance")
        for key, item in test.items():
            if key in mean:
                mean[key] += np.array(item)[:200]
            else:
                mean[key] = np.array(item)[:200]
            testpd[key] = np.array(item)
            if useVariance:
                variancepd[key] = np.array(var[key])

        ARI[names[i]] = testpd.mean(axis=1)

        if useVariance:
            allVariance = variancepd.mean(axis=1)
            plt.plot(ARI[names[i]], label = "gemiddelde")
            plt.plot(ARI[names[i]] - allVariance, alpha = 0.4)
            plt.plot(ARI[names[i]] + allVariance, alpha = 0.4)
            plt.ylim((0.4,1))
            plt.xlabel("#queries")
            plt.ylabel("ARI")
            plt.title(f"Variantie-analyse {names[i]}")
            plt.savefig(f"{location}/variance_{names[i]}.png")
            plt.clf()

    ARI.plot(xlabel="#queries", ylabel="ARI", ylim = (0.4,0.85))

    plt.savefig(f"{location}/ARI.png")

    plt.clf()

    for key, item in mean.items():
        item /= len(paths)

    cbr = []
    for path in paths:
        test = loadDict(path, "total")
        for key, item in test.items():
            cbr.append(np.array(item)[:200] - mean[key])

    cbr = np.array(cbr)

    sorted = np.argsort(cbr, axis = 0)

    all_results = pd.DataFrame()
    indii = np.tile(np.arange(len(paths)*15)[::-1], (200, 1)).T

    for i in range(len(names)):
        indices = np.arange(start=i*15, stop=i*15+15)
        positions = np.isin(sorted,indices)
        all_results[names[i]] = np.where(positions, indii, 0).sum(axis=0) / positions.sum(axis=0)
        
    all_results.plot(xlabel="#queries", ylabel="Aligned rank")
    # plt.show()
    plt.savefig(f"{location}/rank.png")


####################
# Helper functions #
####################

# loadData
def loadDict(path, name):
    path_test = Path(f'{path}/{name}.json').absolute()
    if not os.path.exists(path_test):
        return dict()
    print(f'{path}/{name}.json')
    with open(f'{path}/{name}.json') as json_file:
        # ga verder met een experiment of start
        return json.load(json_file)
    
def saveDict(dict, path, name):
    with open(f"{path}/{name}.json", "w") as outfile:
        json.dump(dict, outfile, indent=4)

def makeFolders(initial, where):
    path = Path(initial).absolute()
    CHECK_FOLDER = os.path.isdir(path)
    if not CHECK_FOLDER:
        os.makedirs(path)
        print("created folder : ", path)

    for place in where:
        path = Path(f"{initial}/{place}").absolute()
        CHECK_FOLDER = os.path.isdir(path)
        if not CHECK_FOLDER:
            os.makedirs(path)
            print("created folder : ", path)

def doAll(path):
    newPath = path.joinpath("plots")
    CHECK_FOLDER = os.path.isdir(newPath)
    if not CHECK_FOLDER:
        os.makedirs(newPath)
        print("created folder : ", newPath)
    makeARI(path, name_algo = "test")
    makeDifferencePlot(path, name_algo = "test")


if __name__ == "__main__":
    def ignore_warnings():
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=Warning)

    ignore_warnings() 


    # test()

    # normalCOBRAS()


    # make plots
    # doAll(Path(f"experimenten/splitlevel4").absolute())

    rank([Path(f"experimenten/thesis/Chapter5/variance_analysis/normalCobras"), 
          Path(f"experimenten/thesis/Chapter5/variance_analysis/splitlevel4")], 
          ["Cobras met dynamisch splitniveau", "Cobras met constant splitniveau = 4"], 
          "experimenten/thesis/Chapter5/variance_analysis", useVariance=True)



    