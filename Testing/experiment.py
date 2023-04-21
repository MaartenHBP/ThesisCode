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
    querylimit = 200
    # runlimit = min(querylimit, len(data))
    runlimit = querylimit


    querier = LabelQuerier(None, target, runlimit)
    clusterer = COBRAS(correct_noise=False, seed=seeds[seed], **arguments)

    all_clusters, _, _, _, = clusterer.fit(data, -1, None, querier)

    if len(all_clusters) < querylimit:
        diff = querylimit - len(all_clusters)
        for ex in range(diff): all_clusters.append(all_clusters[-1])

    return [adjusted_rand_score(target, np.array(clustering)) for clustering in all_clusters]

###############
# Experiments #
###############

def createCOBRAS():
    pass

def initial():
    makeFolders("experimenten", ["initial"])
    learners = [LMNN_wrapper, NCA_wrapper, ITML_wrapper]
    isSupervised = [True, True, False]

    args = {
        "metricLearner" : None,
        "initial" : True,
        "initialSupervised" : 0, # is een percentage
        "initialSemisupervised" : 0,
        "initialRandom" : True, 
    }

    for i, learner in enumerate(learners):
        makeFolders("experimenten/initial", [learner.__name__])
        args["metricLearner"] = learner
        for random in [False, True]:
            makeFolders(f"experimenten/initial/{learner.__name__}", [f"random_{str(random)}"])
            args["initialRandom"] = random

            for k in range(len(relative_testing)):
                supervised = random and isSupervised[i]

                if not supervised:
                    amount = absolute_testing[k]
                    args["initialSemisupervised"] = amount

                else:
                    amount = relative_testing[k]
                    args["initialSupervised"] = amount

                makeFolders(f"experimenten/initial/{learner.__name__}/random_{str(random)}", ['%.1f' % amount])

                path = f"experimenten/initial/{learner.__name__}/random_{str(random)}/{'%.1f' % amount}"
                args["metricLearner"] = learner.__name__
                saveDict(args, path, "settings")
                args["metricLearner"] = learner

                try:
                    with LocalCluster() as cluster, Client(cluster) as client:
                        path_datasets = Path('datasets/cobras-paper/UCI').absolute()
                        datasets = os.listdir(path_datasets)
                        run = dict()
                        p = Path(f'{path}/total.json').absolute()
                        if os.path.exists(p):
                            run = loadDict(path, f"total")
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
                            saveDict(run, path, "total")
                        saveDict(run, path, "total")
                except Exception as x:
                    print("error cccured:" + path)
                    errordict = {"problem": str(x)}
                    saveDict(errordict, path, "error")

def rebuild():
    makeFolders("experimenten", ["rebuild"])

    args = {
        "rebuildPhase": True, 
        "rebuildLevel": "superinstance", # nog niet op superinstance zetten TODO
        "rebuildAmountQueriesAsked" : 75, # gaan hier over loopen
        "rebuildMetric":False, 
        "rebuildSuperInstanceLevel": 0,}


    for k in [50, 75, 100, 125, 150]:
        args["rebuildAmountQueriesAsked"] = k

        path = Path(f"experimenten/rebuild/no_metric/rebuildLevel_{str(args['rebuildLevel'])}/rebuildPartition_False/{str(args['rebuildAmountQueriesAsked'])}").absolute()
        CHECK_FOLDER = os.path.isdir(path)
        if not CHECK_FOLDER:
            os.makedirs(path)
            print("created folder : ", path)
        saveDict(args, path, "settings")

        try:
            with LocalCluster() as cluster, Client(cluster) as client:
                path_datasets = Path('datasets/cobras-paper/UCI').absolute()
                datasets = os.listdir(path_datasets)
                run = dict()
                p = Path(f'{path}/total.json').absolute()
                if os.path.exists(p):
                    run = loadDict(path, f"total")
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
                    saveDict(run, path, "total")
                saveDict(run, path, "total")
        except Exception as x:
            print("error cccured:" + path)
            errordict = {"problem": str(x)}
            saveDict(errordict, path, "error")
                
def test():
    dataset = "glass"
    # args = {
    #     "rebuildPhase": True, 
    #     "rebuildLevel": "all", 
    #     "rebuildAmountQueriesAsked" : 150,
    #     "rebuildMetric": False,
    #     "rebuilder": ClosestVote}

    args = {"after" : True,
        "afterAmountQueriesAsked" : 75,
        "afterMetric" : False, # standaard geen metriek leren
        "afterLevel" : "all",
        "afterSuperInstanceLevel" : 0,
        "afterAllOptions" : False}
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
    plt.plot(runCOBRAS(39, dataset, args), label = "test")
    # args["rebuildMetric"] = True
    # plt.plot(runCOBRAS(16, dataset, args), label = "test_metric")
    plt.plot(runCOBRAS(39, dataset, {"keepSupervised":True}), label = "COBRAS")

    plt.legend()
    

    plt.show()
    
    
######################
# SImple_experiments #
######################
def normalCOBRAS():
    path = Path(f"experimenten/COBRAS").absolute()
    run({}, path)
def rebuilding():
    args = {
        "rebuildPhase": True, 
        "rebuildLevel": "all", 
        "rebuildAmountQueriesAsked" : 100,
        "rebuildMetric": False}


    for k in [True, False]:
        args["rebuildMetric"] = k
        path = Path(f"experimenten/rebuild/metric_{str(args['rebuildMetric'])}/rebuildLevel_{str(args['rebuildLevel'])}/{str(args['rebuildAmountQueriesAsked'])}").absolute()
        run(args, path)

def rebuildingkNN(): # dit is het idee van kNN

    args = {
        "rebuildPhase": True, 
        "rebuildLevel": "all", 
        "rebuildAmountQueriesAsked" : 100,
        "rebuildMetric": False,
        "rebuilder": ClosestVote} # dit werkt via kNN


    for k in [True, False]:
        args["rebuildMetric"] = k
        path = Path(f"experimenten/rebuild_knn/metric_{str(args['rebuildMetric'])}/rebuildLevel_{str(args['rebuildLevel'])}/{str(args['rebuildAmountQueriesAsked'])}").absolute()
        run(args, path)

def rebuildingSuperinstanceLevel():
    args = {
        "rebuildPhase": True, 
        "rebuildLevel": "superinstance",
        "rebuildSuperInstanceLevel": 0,
        "rebuildAmountQueriesAsked" : 100,
        "rebuildMetric": False}


    for k in [True, False]:
        for instlvl in [0, 1, 2]:
            args["rebuildMetric"] = k
            args["rebuildSuperInstanceLevel"] = instlvl
            path = Path(f"experimenten/rebuild/metric_{str(args['rebuildMetric'])}/rebuildLevel_{str(args['rebuildLevel'])}/{str(args['rebuildSuperInstanceLevel'])}/{str(args['rebuildAmountQueriesAsked'])}").absolute()
            run(args, path)
def afterLabelling(): # simpele after labelling
    args = {"after" : True,
        "afterAmountQueriesAsked" : 75,
        "afterMetric" : False, # standaard geen metriek leren
        "afterLevel" : "all",
        "afterSuperInstanceLevel" : 0,
        "afterAllOptions" : False}

    path = Path(f"experimenten/thesis/after/not_all_options").absolute()
    run(args, path) 
def simpleLearning():
    args = {
        "metricAmountQueriesAsked" : 100,
        "learnAMetric" : True}

    path = Path(f"experimenten/learnMetric/{str(args['metricAmountQueriesAsked'])}").absolute()
    run(args, path) 

def simpleLearningCluster():
    args = {
        "metricAmountQueriesAsked" : 100,
        "learnAMetric" : True,
        "metricLevel": "cluster"}

    path = Path(f"experimenten/learnMetric/cluster/{str(args['metricAmountQueriesAsked'])}").absolute()
    run(args, path) 

def simpleLearningSuperinstance():
    args = {
        "metricAmountQueriesAsked" : 100,
        "learnAMetric" : True,
        "metricLevel": "superinstance",
        "metricSuperInstanceLevel": 1}
    
    path = Path(f"experimenten/learnMetric/superinstance1/{str(args['metricAmountQueriesAsked'])}").absolute()
    run(args, path)

def simpleRebuildLearning():
    args = {
        "rebuildPhase": True, 
        "rebuildLevel": "all", 
        "rebuildAmountQueriesAsked" : 100,
        "rebuildMetric": True,
        "rebuilderKeepTransformed": True} # keep the transformed space

    path = Path(f"experimenten/rebuild/metric_{str(args['rebuildMetric'])}_keepMetric/rebuildLevel_{str(args['rebuildLevel'])}/{str(args['rebuildAmountQueriesAsked'])}").absolute()
    run(args, path) 

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
            p = Path(f'{path}/total.json').absolute()
            if os.path.exists(p):
                run = loadDict(path, f"total")
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
                saveDict(run, path, "total")
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

        test_item = np.array(item)[0:200]
        cobras_item = np.array(cobras[key])[0:200]
        
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

def rank(paths): # nog afmaken
    cobras = loadDict(PATH_COBRAS, "total")

    mean = pd.DataFrame()


    for key, item in cobras.items():
        mean[key] = np.array(item)
    
    for path in paths:
        test = loadDict(path, "total")
        for key, item in test.items():
            mean[key] += np.array(item)

    for key, item in mean.items():
        item /= len(paths)

    cbr = []
    for key, item in cobras.items():
        cbr.append(item - mean[key])



    all_results = pd.DataFrame()
    all_results["COBRAS"] = cobraspd.mean(axis=1)
    all_results[name_algo] = testpd.mean(axis=1)

    all_results.plot(xlabel="#queries", ylabel="ARI", ylim = (0,1))
    # plt.show()
    plt.savefig(f"{path}/plots/total.png")


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

    test()

    # afterLabelling()

    # args = {
    #     "metricLearner" : LMNN_wrapper,
    #     "initial" : True,
    #     "initialSupervised" : 0.5, # is een percentage
    #     "initialSemisupervised" : 20,
    #     "initialRandom" : False, 
    # }
    # runCOBRAS(67 ,"hepatitis", arguments=args)

    # test()
    # rebuild()
    # normalCOBRAS()
    # rebuilding() # Al gedaan
    # rebuildingkNN()
    # rebuildingSuperinstanceLevel()
    # simpleLearning()
    # simpleLearningCluster()
    # simpleLearningSuperinstance()
    # simpleRebuildLearning()

    # make plots
    # doAll(Path(f"experimenten/thesis/after/not_all_options").absolute())

    # doAll(Path(f"experimenten/rebuild/metric_True/rebuildLevel_superinstance/0/100").absolute())

    # doAll(Path(f"experimenten/rebuild/metric_False/rebuildLevel_superinstance/0/100").absolute())

    # path = Path(f"experimenten/rebuild/metric_False/rebuildLevel_superinstance/0/100").absolute()
    # makeARI(path, name_algo = "test")
    # makeDifferencePlot(path, name_algo = "test")


    