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

# moonplots
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

from sklearn.model_selection import train_test_split

plt.style.use('default')

nbRUNS = 100
ARGUMENTS = range(25) # 25 seeds voor betere curve mar geen overkill aan runs
SEED = 24
random_generator = np.random.default_rng(SEED)
seeds = [random_generator.integers(1,1000000) for i in range(nbRUNS)] # creation of the seeds

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

    querylimit = 200
    runlimit = querylimit


    querier = LabelQuerier(None, target, runlimit)
    clusterer = COBRAS(correct_noise=False, seed=seeds[seed], **arguments)

    all_clusters, _, _, _, = clusterer.fit(data, -1, None, querier)

    if len(all_clusters) < querylimit:
        diff = querylimit - len(all_clusters)
        for ex in range(diff): all_clusters.append(all_clusters[-1])

    return [adjusted_rand_score(target, np.array(clustering)) for clustering in all_clusters]

def kNNTest(seed, dataName, metric_algo: MetricLearner):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    path = Path(f'datasets/cobras-paper/UCI/{dataName}.data').absolute()
    dataset = np.loadtxt(path, delimiter=',')
    data = dataset[:, 1:]
    target = dataset[:, 0]

    querylimit = 200
    runlimit = querylimit


    querier = LabelQuerier(None, target, runlimit)
    clusterer = COBRAS(correct_noise=False, seed=seeds[seed], **{"useNewConstraintIndex" : True,
        "mergeBlobs" : True})

    all_clusters, _, _, _, = clusterer.fit(data, -1, None, querier)

    labeled = clusterer.constraint_index_advanced.labeled

    result = []

    for j in [25, 50, 75, 100]:
        i = j
        if len(labeled) < i:
            i = len(labeled) 
        indi = np.array(labeled)[np.arange(i)]
        newData = metric_algo(np.copy(data)).fit_transform(None, None, indi, np.copy(target)[indi])
        model = KNeighborsClassifier(n_neighbors=3, weights='distance')
        model.fit(np.array(newData)[indi], np.copy(target)[indi])
        prediction = model.predict(np.array(newData))

        mask = np.ones(data.shape[0], dtype=bool)
        mask[indi] = False

        result.append(adjusted_rand_score(target[mask], prediction[mask]))

    return result

def count(seed, dataName, arguments):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    path = Path(f'datasets/cobras-paper/UCI/{dataName}.data').absolute()
    dataset = np.loadtxt(path, delimiter=',')
    data = dataset[:, 1:]
    target = dataset[:, 0]

    querylimit = 200
    runlimit = querylimit


    querier = LabelQuerier(None, target, runlimit)
    clusterer = COBRAS(correct_noise=False, seed=seeds[seed], **arguments)

    all_clusters, _, _, _, = clusterer.fit(data, -1, None, querier)

    lan = clusterer.countLabelled

    if len(all_clusters) < querylimit:
        diff = querylimit - len(all_clusters)
        for ex in range(diff): lan.append(lan[-1])

    return lan # return the counts


    

###############
# Experiments #
###############
                
def test():
    dataset = "breast-cancer-wisconsin"

    args = {   
        
        "splitlevel_strategy": None,
        "splitlevelInt" : 4,
        
        "metricLearner" : "KLMNN_wrapper",
        "metricLearer_arguments" : {},
        "changeToMedoids": False,
        "cluster_algo": "KMeansClusterAlgorithm", 
        "new_split": True,

        "metricLevel" : "all",
        "metricSuperInstanceLevel" : 0,

        "learnAMetric" : False,
        "metricAmountQueriesAsked" : 50,
        "metricInterval" : 0,

        "initial" : False,
        "initialSupervised" : 0.5, 
        "initialRandom" : True, 

        "rebuildPhase" : False,
        "rebuildAmountQueriesAsked" : 75,
        "rebuildInterval" : 0,
        "rebuildLevel" : "all", 
        "rebuildSuperInstanceLevel" : 0,
        "rebuilder" : "ClosestVote",
        "rebuildMetric" : False,
        "rebuilderKeepTransformed" : False,
        "rebuildCluster": True,

        "after" : False,
        "afterAmountQueriesAsked" : 1,
        "after_k" : 3,
        "after_weights" : "distance",
        "afterMetric" : False, 
        "afterKeepTransformed" : False, 
        "afterLevel" : "all", 
        "afterSuperInstanceLevel" : 3,
        "afterRadius" : False,
        "afterLambda" : 1,

        "useNewConstraintIndex" : True,
        "mergeBlobs" : True, 
        "represBlobs" : False
    }

    

    # plt.show()    
    # plt.plot(runCOBRAS(9, dataset, args), label = "kNN")
    # print("next")
    # args["afterRadius"] = True
    # plt.plot(runCOBRAS(10, dataset, args), label = "ITML")
    # print("next")
    plt.plot(runCOBRAS(4, dataset, args), label = "distance")
    # print("next")
    # args["rebuildCluster"] = False
    # plt.plot(runCOBRAS(17, dataset, args), label = "baseline")
    # print("next")
    # args["rebuildPhase"] = False
    # plt.plot(runCOBRAS(17, dataset, args), label = "COBRAS")
    # # plt.plot(runCOBRAS(16, dataset, args), label = "test_metric")
    # # args["rebuildMetric"] = True
    # # plt.plot(runCOBRAS(16, dataset, args), label = "test_metric")
    # # plt.plot(runCOBRAS(20, dataset, {"useNewConstraintIndex" : True, "mergeBlobs" : True}), label = "COBRASLabels")
    # # print("next")

    # args["useNewConstraintIndex"] = True
    # args["plusBlobs"] = True
    # plt.plot(runCOBRAS(16, dataset, args), label = "COBRAS+")
    # # plt.plot(runCOBRAS(16, dataset, {}), label = "COBRAS")

    plt.legend()
    

    plt.show()

def moonPlot():
    # features, true_labels = make_moons(n_samples=100, noise=0.13)

    for i in ["oorspronkelijk", "kmeans", "spectral", "ITML", "NCA", "KLMNN", "GBLMNN", "LMNN"]:

        print(i)


        path = Path(f'datasets/created/spectral.data').absolute()
        dataset = np.loadtxt(path, delimiter=',')
        features = dataset[:, 1:]
        true_labels = dataset[:, 0]

        X_train, X_test, y_train, y_test = train_test_split(np.arange(len(features)), true_labels, test_size=0.5, random_state=42)

        if i == "kmeans":
            kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
            true_labels = kmeans.labels_

        if i == "spectral":
            clustering = SpectralClustering(n_clusters=2, eigen_solver="arpack",
                affinity="nearest_neighbors",).fit(features)
            true_labels = clustering.labels_

        if i == "ITML":
            features = ITML_wrapper(features).fit_transform(None, None, X_train, y_train)

        if i == "NCA":  
            features = NCA_wrapper(features).fit_transform(None, None, X_train, y_train)

        if i == "LMNN":  
            features = LMNN_wrapper(features).fit_transform(None, None, X_train, y_train)

        if i  == "KLMNN":
            features = KLMNN_wrapper(features).fit_transform(None, None, X_train, y_train)

        if i == "GBLMNN":
            features = GBLMNN_wrapper(features).fit_transform(None, None, X_train, y_train)

        if i == "GB_LMNN":
            features = KLMNN_wrapper(features).fit_transform(None, None, np.arange(len(features)), true_labels)
            plt.scatter(features[:,0], features[:,1], c=true_labels)
            plt.show()
            features = GBLMNN_wrapper(features).fit_transform(None, None, np.arange(len(features)), true_labels)


        # GBLMNN

        plt.scatter(features[X_train,0], features[X_train,1], c=y_train)
        plt.scatter(features[X_test,0], features[X_test,1], c=y_test, alpha=0.3)
        plt.savefig(f"experimenten/thesis/2-literatuurstudie/moons_0.5/{i}.png", dpi = 600)
        plt.clf()

def subjective_dataset():
    features, true_labels = make_blobs(
    n_samples=500, cluster_std=[1, 2, 2], random_state=33)


    # kmeans = KMeans(n_clusters=3, random_state=0).fit(features)
    # true_labels = kmeans.labels_

    clustering = SpectralClustering(n_clusters=3, eigen_solver="arpack",
        affinity="nearest_neighbors",).fit(features)
    true_labels = clustering.labels_

    plt.scatter(features[:,0], features[:,1], c=true_labels)
    plt.savefig(f"experimenten/thesis/1-Introductie/spectral_3.png", dpi = 600)


    
    
######################
# Simple_experiments #
######################
def normalCOBRAS():
    path = Path(f"experimenten/thesis/Chapter_rebuild/closest/metric").absolute()
    run({ "useNewConstraintIndex" : True,
          "splitlevel_strategy": "constant",
        "splitlevelInt" : 4
        # "mergeBlobs" : True,
        # "rebuildPhase": True,
        # "rebuildAmountQueriesAsked" : 100,
        # "rebuildAllOptions": True,
        # "rebuilder" : 'Vote',
        # "rebuildMetric" : True
        }, path)


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
            for j in range(len(datasets)):
                nameData = datasets[j][:len(datasets[j]) - 5]
                if nameData in run:
                    continue
                print(f"({path})\t ({nameData})\t Running")
                parallel_func = functools.partial(runCOBRAS, dataName = nameData, arguments = args)
                futures = client.map(parallel_func, ARGUMENTS)
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

def makeARI(path, name_algo): # momenteel enkel vergelijken met COBRAS, en ook nog enkel absolute: TODO
    test = loadDict(path[0], "total")
    test1 = loadDict(path[1], "total")
    cobras = loadDict("experimenten/thesis/5-Labels/label_blobs_test/COBRASD++M", "total")
    # cobraspd = pd.DataFrame()

    


    for key, item in test.items():

        
        plt.plot(np.array(cobras[key]), label = "COBRAS")
        plt.plot(np.array(item), label = name_algo[0])
        plt.plot(np.array(test1[key]), label = name_algo[1])

        plt.title(key)
        plt.xlabel("#vragen")
        plt.ylabel("ARI")
        # plt.legend()
        print("hier")
        plt.savefig(f"{path[0]}/plots/{key}.png", dpi = 600)
        plt.clf()

    # all_results = pd.DataFrame()
    # all_results["COBRAS"] = cobraspd.mean(axis=1)
    # all_results[name_algo] = testpd.mean(axis=1)

    # all_results.plot(xlabel="#queries", ylabel="ARI", ylim = (0.4,0.85))
    # # plt.show()
    # plt.savefig(f"{path}/plots/total.png")

    # plt.clf()

def makeDifferencePlot(paths, name_algos = ""):

    for i in range(len(paths)):
        path = paths[i]
        name_algo = name_algos[i]
        test = loadDict(path, "total")
        cobras = loadDict("experimenten/thesis/5-Labels/label_blobs_test/COBRASD++M", "total")
        
        total = np.zeros(200)

        if not name_algo:
            name_algo = path


        for key, item in test.items():

            test_item = np.array(item)[:200]
            cobras_item = np.array(cobras[key])[:200]
            
            bools = test_item > cobras_item

            total += bools

        
        plt.plot(total, label = name_algo)

    plt.xlabel("#vragen")
    plt.ylabel("#ARI>COBRAS")
    plt.ylim((0, 15))
    plt.legend()
    plt.savefig(f"experimenten/thesis/7-kNN/better/all.png", dpi=600)

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
            CHECK_FOLDER = os.path.isdir(os.path.join(location, "variance"))
            if not CHECK_FOLDER:
                os.makedirs(os.path.join(location, "variance"))
                print("created folder : ", os.path.join(location, "variance"))
            allVariance = variancepd.mean(axis=1)
            plt.plot(ARI[names[i]], label = "gemiddelde")
            plt.plot(ARI[names[i]] - allVariance, alpha = 0.4, label = "-$\sigma$")
            plt.plot(ARI[names[i]] + allVariance, alpha = 0.4, label = "+$\sigma$")
            plt.ylim((0.4,1))
            plt.xlabel("#vragen")
            plt.ylabel("Gemiddelde ARI")
            # plt.title(f"Variantie-analyse {names[i]}")
            plt.legend()
            plt.savefig(f"{location}/variance/variance_{names[i]}.png", dpi = 600)
            plt.clf()

    ARI.plot(xlabel="#vragen", ylabel="Gemiddelde ARI", ylim = (0.4,0.85))

    plt.savefig(f"{location}/ARI.png", dpi = 600)

    # plt.show()

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
        
    all_results.plot(xlabel="#vragen", ylabel="AR")
    # plt.show()
    plt.savefig(f"{location}/rank.png", dpi = 600)

    plt.clf()

def rank_kNN(paths, names, location):
    # cobras = loadDict(PATH_COBRAS, "total")

    mean = pd.DataFrame()


    ARI = pd.DataFrame()


    # for key, item in cobras.items():
    #     mean[key] = np.array(item)
    
    for i in range(len(paths)):
        path = paths[i]
        testpd = pd.DataFrame()
        test = loadDict(path, "total")
        for key, item in test.items():
            if key in mean:
                mean[key] += np.array(item)
            else:
                mean[key] = np.array(item)
            testpd[key] = np.array(item)

        ARI[names[i]] = testpd.mean(axis=1)

    ARI.to_csv(f"{location}/ARI.csv", index=False)


    for key, item in mean.items():
        item /= len(paths)

    cbr = []
    for path in paths:
        test = loadDict(path, "total")
        for key, item in test.items():
            cbr.append(np.array(item) - mean[key])

    cbr = np.array(cbr)

    sorted = np.argsort(cbr, axis = 0)

    all_results = pd.DataFrame()
    indii = np.tile(np.arange(len(paths)*15)[::-1], (4, 1)).T

    for i in range(len(names)):
        indices = np.arange(start=i*15, stop=i*15+15)
        positions = np.isin(sorted,indices)
        all_results[names[i]] = np.where(positions, indii, 0).sum(axis=0) / positions.sum(axis=0)
        
    all_results.plot(xlabel="#queries", ylabel="Aligned rank")
    # plt.show()
    all_results.to_csv(f"{location}/aligned.csv", index=False)

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

#########################
# Run alle experimenten #
#########################
def runAll(doAll = False):
    rootdir = Path(f"queue").absolute()

    for subdir in os.listdir(rootdir):
        # Komen nu bij een chapter
        file_chapter = Path(f"experimenten/thesis/{subdir}").absolute()
        chapter_location = os.path.join(rootdir, subdir)
        print(f"({file_chapter})")

        for subchapterdir in os.listdir(chapter_location):
            # komen nu bij het soort experiment
            file_experiment = os.path.join(file_chapter, subchapterdir)
            experiment_location = os.path.join(chapter_location, subchapterdir)
            print(f"({file_experiment})")

            all_paths  = []
            all_names = []

            found = 0

            for file in os.listdir(experiment_location):
                results_location = os.path.join(file_experiment, file[:len(file) - 5])

                print(f"({results_location})")
                

                experiment_file = loadDict(experiment_location, file[:len(file) - 5]) 
                all_names.append(experiment_file["plotName"])

                if "reuse" in experiment_file:
                    reuse = Path(f"experimenten/thesis/{experiment_file['reuse']}/total.json").absolute()
                    if reuse.is_file():
                        all_paths.append(Path(f"experimenten/thesis/{experiment_file['reuse']}").absolute())
                        found += 1
                        continue # de resultaten zijn hier al van bekend
                
                all_paths.append(results_location)

                test =  Path(os.path.join(results_location, "total.json")).absolute()
                if test.is_file():
                    found += 1
                    continue

                run(experiment_file["settings"], results_location)

            if doAll and found == len(os.listdir(experiment_location)):
                continue
            rank(all_paths, 
            all_names, 
            file_experiment, useVariance=True)

def kNNExperiment():
    all_paths = []
    arry = ["ITML_wrapper", "NCA_wrapper", "LMNN_wrapper", "KLMNN_wrapper",
              "GBLMNN_wrapper", "GB_LMNN_wrapper", "EUCLIDIAN_wrapper", 
              "KLMNN_poly1_wrapper", "KLMNN_poly2_wrapper", 
              "KLMNN_poly3_wrapper" ,"KLMNN_rbf_wrapper", "LMNN_wrapper_2"]
    for i in arry:
        path = Path(f"experimenten/thesis/7-kNN/metric_test/{i}").absolute()
        all_paths.append(path)

        CHECK_FOLDER = os.path.isdir(path)
        if not CHECK_FOLDER:
            os.makedirs(path)
            print("created folder : ", path)
        try:
            print(f"Started {i}")
            with LocalCluster() as cluster, Client(cluster) as client:
                path_datasets = Path('datasets/cobras-paper/UCI').absolute()
                datasets = os.listdir(path_datasets)
                run = dict()
                p = Path(f'{path}/total.json').absolute()
                if os.path.exists(p):
                    run = loadDict(path, f"total")
                for j in range(len(datasets)):
                    nameData = datasets[j][:len(datasets[j]) - 5]
                    if nameData in run:
                        continue
                    print(f"({path})\t ({nameData})\t Running")
                    parallel_func = functools.partial(kNNTest, dataName = nameData, metric_algo = eval(i))
                    futures = client.map(parallel_func, ARGUMENTS)
                    results = np.array(client.gather(futures))
                    run[nameData] = np.mean(results, axis=0).tolist()
                    saveDict(run, path, "total")
                saveDict(run, path, "total")
        except Exception as x:
            print("error cccured:" + path)
            errordict = {"problem": str(x)}
            saveDict(errordict, path, "error")
    rank_kNN(all_paths, arry, "experimenten/thesis/7-kNN/metric_test")

def labelledCount():
    args = {"useNewConstraintIndex" : True,
        "mergeBlobs" : True, 
        "represBlobs" : False}
    
    total = pd.DataFrame()

    for i in [True, False]:
        args["mergeBlobs"] = i
        args["represBlobs"] = not i

        print(args["mergeBlobs"] )
        print(args["represBlobs"])
        with LocalCluster() as cluster, Client(cluster) as client:
            path_datasets = Path('datasets/cobras-paper/UCI').absolute()
            datasets = os.listdir(path_datasets)
            run = np.zeros(200)
            for j in range(len(datasets)):
                nameData = datasets[j][:len(datasets[j]) - 5]
                if nameData in run:
                    continue
                parallel_func = functools.partial(count, dataName = nameData, arguments = args)
                futures = client.map(parallel_func, ARGUMENTS)
                results = np.array(client.gather(futures))
                run += np.mean(results, axis=0).tolist()
            
            if i:
                total["Complete graaf"] = run/15
            else:
                total["Incomplete graaf"] = run/15

    perc = (np.array(total["Complete graaf"]) - np.array(total["Incomplete graaf"]))/np.array(total["Incomplete graaf"])

    print(np.mean(perc))


    total.plot(xlabel="#vragen", ylabel="Gemiddeld #constraints")
    # plt.show()
    plt.savefig(f"experimenten/thesis/count.png", dpi = 600)

            

if __name__ == "__main__":
    def ignore_warnings():
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=Warning)

    ignore_warnings() 

    runAll(doAll = False) # vanaf nu dit oproepen

    # test()

    ################
    # Create plots #
    ################

    lis = ["experimenten/thesis/7-kNN/radius_KLMNN/COBRASD++M_kNN_radius",
           "experimenten/thesis/7-kNN/radius_KLMNN/COBRASD++M_kNN_radius_KLMNN"]
    
    names = ["rNN", "rNN-KLMNN"]

    # lis = ["experimenten/thesis/8-rebuild/2-split-after/COBRASD++M _split",
    #        "experimenten/thesis/7-kNN/radius_KLMNN/COBRASD++M_kNN_radius"]
    
    # names = ["split", "COBRAS"]

    # makeARI(lis, name_algo = names)

    # for i in range(len(lis)):
    # makeARI(lis, names)

    # makeARI("experimenten/thesis/6-transformaties/NCA/COBRASD++M_metric_075")

    # labelledCount()

    # kNNTest(10, "ecoli", ITML_wrapper)


    # moonPlot()

    # kNNExperiment()

    # normalCOBRAS()


    # make plots
    # doAll(Path(f"experimenten/thesis/posterevent/kNN_metric").absolute())

    #################
    # Variance_test #
    #################

    # # rank([Path(f"experimenten/thesis/4-COBRAS/variance_analysis/normalCOBRAS"),
    #       Path(f"experimenten/thesis/4-COBRAS/variance_analysis/splitlevel4"),], 
    #       ["dynamisch splitniveau", "splitniveau = 4"], 
    #       "experimenten/thesis/4-COBRAS/variance_analysis", useVariance=True)









    