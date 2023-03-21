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
# from metric_learn import * -> werken met wrappers

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
seedsForTSNE = [1, 25, 46, 87]

ABSOLUTE = 200
RELATIVE = 0.3

ABSOLUTE_TEST = [30, 50, 80, 120, 170]
RELATIVE_TEST = [0.1, 0.2, 0.4, 0.6, 0.9] # percentage of the constriaints it has

# eerste testen werken enkel met NCA en daarna met ITML en NCA+ITML (keep it simple)

#################
# Parallel code #
#################

def runCOBRAS(seed, dataName):
    """This is the code to be called for the running COBRAS:
        - COBRAS 
        - Saving the repres with their clustering label and constraints for later use
    """
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    path = Path(f'datasets/cobras-paper/UCI/{dataName}.data').absolute()
    dataset = np.loadtxt(path, delimiter=',')
    data = dataset[:, 1:]
    target = dataset[:, 0]

    querylimit = max(math.floor(len(data)*RELATIVE), ABSOLUTE)

    querier = LabelQuerier(None, target, querylimit)
    clusterer = COBRAS(correct_noise=False, seed=seeds[seed])

    all_clusters, _, _, repres, _, _, all_constraints = clusterer.fit(data, -1, None, querier)

    def labelRepres(v,k):
        return np.array(all_clusters[k])[np.array(v)].tolist()
    
    repres_labels = {k:labelRepres(v,k) for k,v in repres.items()}


    saveNumpy("experimenten/COBRAS", [all_constraints], dataName, ["all_constraints"], seed)
    saveDicts("experimenten/COBRAS", [repres_labels, repres], dataName, ["repres_labels", "repres"], seed)

    if len(all_clusters) < querylimit:
        diff = querylimit - len(all_clusters)
        for ex in range(diff): all_clusters.append(all_clusters[-1])

    return [adjusted_rand_score(target, np.array(clustering)) for clustering in all_clusters]

def Experiment1(seed, dataName, metricLearner, expand, index, relative = False, random = False): # momenteel alleen met semisupervised
    """This is the code to be called for the first experiment:
        - learn a metric beforehand using piarzise constraints and then run COBRAS, ask randomconstraints already to COBRAS 
        - for seedsForTSNE calculate the TSNE, to see what the transformation does
    """
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    path = Path(f'datasets/cobras-paper/UCI/{dataName}.data').absolute()
    dataset = np.loadtxt(path, delimiter=',')
    data = dataset[:, 1:]
    target = dataset[:, 0]

    querylimit = max(math.floor(len(data)*RELATIVE), ABSOLUTE)

    querier = LabelQuerier(None, target, querylimit)
    clusterer = COBRAS(correct_noise=False, seed=seeds[seed])
    clusterer.tokenFit(data, -1, None, querier)

    # first learn the transformation
    amount = math.floor(math.floor(RELATIVE_TEST[index] * math.floor(len(data)*RELATIVE))) if relative else ABSOLUTE_TEST[index]
    path = Path(f'experimenten/COBRAS/all_constraints/{dataName}/{seed}.data').absolute()
    pairs, constraints = (np.loadtxt(path, delimiter=',', dtype=int)[:, :2], np.loadtxt(path, delimiter=',', dtype=int)[:, 2]) if not random else clusterer.query_random_points(options=np.arange(len(data)),count = amount)
    

    metric = metricLearner(preprocessor = np.copy(data), seed = seeds[seed], expand = expand)
    newData = metric.fit_transform(pairs[:amount], constraints[:amount], None, None)

    if seed in seedsForTSNE:
        path_TSNE = Path(f'experimenten/TSNE/{dataName}_{seed}_{random}_{amount}').absolute()
        data_TSNE = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(newData)
        np.savetxt(path_TSNE, data_TSNE, delimiter=',')  
    # choose the constraints and make them

    all_clusters, _, _, _, _, _, _ = clusterer.fit(newData, -1, None, querier)

    if len(all_clusters) < querylimit:
        diff = querylimit - len(all_clusters)
        for ex in range(diff): all_clusters.append(all_clusters[-1])

    return [adjusted_rand_score(target, np.array(clustering)) for clustering in all_clusters]

def Experiment2(seed, dataName, metricLearner, metriclearner_arguments, function, functionarguments, askExtraConstraints = 0): # function is hoe je de nieuwe clustering maakt (hier zijn een paar ideeen mogelijk)
    """This is the code to be called for the second experiment:
        - learn a metric to make the results of COBRAS better afterwards
        - for seedsForTSNE calculate the TSNE, to see what the transformation does
    """
    pass

def Experiment3(seed, dataName, metricLearner, metriclearner_arguments): 
    """This is the code to be called for the third experiment:
        - learn a metric during the run of COBRAS
    """
    pass

def Experiment4(seed, dataName, keep = False):
    """This is the code to be called for the fourth experiment: 
        - Own idea
    """
    pass

def Experiment5(seed, dataName, keep = False):
    """This is the code to be called for the fifth experiment: 
        - Initial Spectral clustering
    """
    pass

###############
# Experiments #
###############

def normalCOBRAS():
    print("==Running COBRAS==")
    makeFolders("experimenten/COBRAS", ["repres_labels", "repres", "all_constraints", "plots"])
    makeFolders("experimenten/COBRAS/plots", ["absolute", "relative_points", "animations"])
    path_datasets = Path('datasets/cobras-paper/UCI').absolute()
    datasets = os.listdir(path_datasets)
    cobras = dict()
    ##########################################################
    with LocalCluster() as cluster, Client(cluster) as client:
        for j in range(len(datasets)):
            nameData = datasets[j][:len(datasets[j]) - 5]
            for k in ["repres_labels", "repres", "all_constraints", "plots/absolute", "plots/relative_points", "plots/animations"]:
                makeFolders(f"experimenten/COBRAS/{k}", [nameData])
            doTSNE(nameData)
            print(f"(COBRAS) ({nameData})\t Running")
            parallel_func = functools.partial(runCOBRAS, dataName = nameData)
            futures = client.map(parallel_func, ARGUMENTS)
            results = np.array(client.gather(futures))
            cobras[nameData] = np.mean(results, axis=0).tolist()
    print(f"(COBRAS)\t Saving results")
    
    saveDict(cobras, f"experimenten/COBRAS", "total")


def initialTransformation():
    print("==Running Initial transformation==")
    makeFolders("experimenten", ["initial"])
    for expand in [True, False]:
        for random in [True, False]:
            for index in range(len(ABSOLUTE_TEST)):
                for relative in [True, False]:
                    exp1(ITML_wrapper, expand, index, random, relative)
    
    
    ##########################################################

def exp1(metricLearner, expand, index, random, relative):
    path = Path(f"experimenten/initial/total_{metricLearner.__name__})_expand{str(expand)}_random{str(random)}_index{str(index)})_(relatitve{str(relative)}).json")
    if  os.path.exists(path):
        return
    print(f"(Experiment1) ({metricLearner.__name__}) (expand:{str(expand)}) (random:{str(random)}) (index:{str(index)}) (relatitve:{str(relative)})\t Running")
    with LocalCluster() as cluster, Client(cluster) as client:
        path_datasets = Path('datasets/cobras-paper/UCI').absolute()
        datasets = os.listdir(path_datasets)
        cobras = dict()
        for j in range(len(datasets)):
            nameData = datasets[j][:len(datasets[j]) - 5]
            print(f"(Experiment1) ({metricLearner.__name__}) (expand:{str(expand)}) (random:{str(random)}) (index:{str(index)}) ({nameData})\t Running")
            parallel_func = functools.partial(Experiment1, dataName = nameData, metricLearner = metricLearner, expand = expand, index = index, random = random, relative = relative)
            futures = client.map(parallel_func, ARGUMENTS)
            results = np.array(client.gather(futures))
            cobras[nameData] = np.mean(results, axis=0).tolist()
    print(f"(Experiment1)\t Saving results")
    
    saveDict(cobras, f"experimenten/initial", f"total_{metricLearner.__name__})_expand{str(expand)}_random{str(random)}_index{str(index)})_(relatitve{str(relative)})")

####################
# Helper functions #
####################

### TSNE ###
def doTSNE(dataName):   
    path = Path(f'datasets/cobras-paper/UCI/{dataName}.data').absolute()
    dataset = np.loadtxt(path, delimiter=',')
    data = dataset[:, 1:]
    makeFolders("experimenten/", ["TSNE"])
    path = Path(f'experimenten/TSNE/{dataName}.data').absolute()
    if  os.path.exists(path):
        return 
    print(f"(TSNE) ({dataName})\t Running")
    newData = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(data)
    np.savetxt(path, newData, delimiter=',')   


### saving data ###

def saveNumpy(initial, numpyArray, dataName, where, seed):
    for array, place in zip(numpyArray, where):
        path = Path(f'{initial}/{place}/{dataName}/{seed}.data').absolute()
        np.savetxt(path, array, delimiter=',')

def saveDicts(initial, dict, dataName, where, seed): # for multiple dictionaries
    for array, place in zip(dict, where):
        with open(f'{initial}/{place}/{dataName}/{seed}.json', "w") as outfile:
            json.dump(array, outfile, indent=4)

def saveDict(dict, path, name):
    with open(f"{path}/{name}.json", "w") as outfile:
        json.dump(dict, outfile, indent=4)

# loadData
def loadDict(path, name):
    path_test = Path(f'{path}/{name}.json').absolute()
    if not os.path.exists(path_test):
        return dict()
    print(f'{path}/{name}.json')
    with open(f'{path}/{name}.json') as json_file:
        # ga verder met een experiment of start
        return json.load(json_file)

### Folders ###

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
           
if __name__ == "__main__":
    def ignore_warnings():
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=Warning)

    ignore_warnings() # moet meegegeven worden met de workers tho
    # normalCOBRAS()
    # runCOBRAS(19,"breast-cancer-wisconsin")
    # Experiment1(25, "parkinsons", ITML_wrapper, {}, 2, random = True)
    initialTransformation()