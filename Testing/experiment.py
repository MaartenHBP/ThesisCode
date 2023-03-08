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
from noise_robust_cobras.metric_learning.metriclearning_algorithms import *
from noise_robust_cobras.metric_learning.rebuildInstance import *
from noise_robust_cobras.metric_learning.metriclearning import *
from noise_robust_cobras.clustering_algorithms.clustering_algorithms import *
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

nbRUNS = 100
ARGUMENTS = range(100)
SEED = 24
random_generator = np.random.default_rng(SEED)
seeds = [random_generator.integers(1,1000000) for i in range(nbRUNS)] # creqtion of the seeds
QUERYLIMIT = 200
BASELINELIMIT = 550

def runAlgo(seed, dataName, parameters, querylimit = 200):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)

    path = Path(f'datasets/cobras-paper/UCI/{dataName}.data').absolute()
    dataset = np.loadtxt(path, delimiter=',')
    data = dataset[:, 1:]
    target = dataset[:, 0]

    # print(f"classlabels = {len(np.unique(target))}")
    clusterer = COBRAS(correct_noise=False, seed=seeds[seed], **parameters)
    if not clusterer.baseline:
        querier = LabelQuerier(None, target, querylimit)
    else:
        querier = LabelQuerier(None, target, BASELINELIMIT)
    # try:
    if clusterer.baseline: # sketchy baseline
        all_clusters, runtimes, *_ = clusterer.fit(data, -1, None, querier)
        data = np.copy(clusterer.data)
        querier = LabelQuerier(None, target, querylimit)
        clusterer = COBRAS(correct_noise=False, seed=seeds[seed]) # run standard cobras


    all_clusters, runtimes, *_ = clusterer.fit(data, -1, None, querier)
    if len(all_clusters) < QUERYLIMIT:
        diff = QUERYLIMIT - len(all_clusters)
        for ex in range(diff): all_clusters.append(all_clusters[-1])
    return [adjusted_rand_score(target, np.array(clustering)) for clustering in all_clusters]
    # except Exception as e: # failed equals zero results
    #     print(e)
    #     return [np.zeros(len(data)) for i in range(QUERYLIMIT)]


def run(EXPERIMENT_PATH, queue):
    with LocalCluster() as cluster, Client(cluster) as client:

        ##########################################################
        # gaan ervan uit dat alles in de queue van een experiment is
        path = Path(EXPERIMENT_PATH).absolute()
        CHECK_FOLDER = os.path.isdir(path)
        if not CHECK_FOLDER:
            os.makedirs(path)
            print("created folder : ", path)
        ##########################################################

        ##########################################################
        # all the results
        all_results = pd.DataFrame()
        path_results = Path(f'{EXPERIMENT_PATH}/results').absolute()
        if os.path.exists(path_results):
            all_results = pd.read_csv(path_results, index_col=0)
        ##########################################################

        #########################################################
        # run normal COBRAS first
        cobras = None # the results drom COBRAS
        path_cobras = Path(f"experimenten/COBRAS").absolute()
        if os.path.exists(path_cobras):
            cobras = pd.read_csv(path_cobras, index_col=0)
            all_results["Cobras"] = cobras["total"]
            all_results.to_csv(path_results)
            # print(all_results)
        else:
            path_datasets = Path('datasets/cobras-paper/UCI').absolute()
            datasets = os.listdir(path_datasets)
            cobras = pd.DataFrame()
            for j in range(len(datasets)):
                nameData = datasets[j][:len(datasets[j]) - 5]
                print(f"Started running COBRAS on {nameData}")
                parallel_func = functools.partial(runAlgo, dataName = nameData, parameters = {})
                futures = client.map(parallel_func, ARGUMENTS)   
                results = np.array(client.gather(futures))
                cobras[nameData] = np.mean(results, axis=0)
                print(f"Finished running COBRAS")
            cm = cobras.mean(axis=1)
            cobras["total"] = cm
            all_results["Cobras"] = cm
            all_results.to_csv(path_results)
            cobras.to_csv(f"experimenten/COBRAS")
        ##########################################################


        ##########################################################
        path_data = Path(queue).absolute()
        dir_list = os.listdir(path_data)
        for k in range(len(dir_list)):
            i = dir_list[k][:len(dir_list[k]) - 5]
            with open(f'{queue}/{dir_list[k]}') as json_file:
                print(f"Start/continue {i}")
                # ga verder met een experiment of start
                experiments = json.load(json_file)

                # get the needed parameters
                experiment = experiments["cobrasparam"]

                # setup the arguments
                try:
                    settings = dict() # is not a deepcopy if you have nested disctionaries (see documentation)
                    if "metric" in experiment:
                        metricsettings = copy.deepcopy(experiment["metric_parameters"])
                        if "metric" in metricsettings:
                            metricsettings["metric"]["value"] = eval(metricsettings["metric"]["value"])
                        settings["metric"] = eval(experiment["metric"]) if experiment["metric"] else None
                        settings["metric_parameters"] = metricsettings
                    if "cluster_algo" in experiment:
                        settings["cluster_algo"] = eval(experiment["cluster_algo"]) if experiment["cluster_algo"] else None
                        settings["cluster_algo_parameters"] = experiment["cluster_algo_parameters"]

                    if "rebuild_cluster" in experiment:
                        settings["rebuild_cluster"] = eval(experiment["rebuild_cluster"]) if experiment["rebuild_cluster"] else None
                        settings["rebuild_cluster_parameters"] = experiment["rebuild_cluster_parameters"]

                    if "rebuilder" in experiment:
                        settings["rebuilder"] = eval(experiment["rebuilder"]) if experiment["rebuilder"] else None
                        settings["rebuilder_parameters"] = experiment["rebuilder_parameters"]

                    if "baseline" in experiment:
                        settings["baseline"] = experiment["baseline"]
                except Exception as e: # failed equals zero results
                    print(e)
                    continue

                # make path for results of this run
                path_exp = f'{EXPERIMENT_PATH}/{i}'
                path = Path(path_exp).absolute()
                CHECK_FOLDER = os.path.isdir(path)
                if not CHECK_FOLDER:
                    os.makedirs(path)
                    print("created folder : ", path)

                pathMean = Path(f'{path_exp}/ARI')
                pathTimestamps = Path(f'{path_exp}/timestamps')
                mean = pd.DataFrame()
                timestamps = pd.DataFrame()
                if os.path.exists(pathMean):
                    mean = pd.read_csv(pathMean, index_col=0)
                    timestamps = pd.read_csv(pathTimestamps, index_col=0)
                # dump the settings of this task also in the resulting folder
                with open(f"{path_exp}/settings.json", "w") as outfile:
                    json.dump(experiments, outfile, indent=4)
                
                # go over all the UCI datasets
                path_datasets = Path('datasets/cobras-paper/UCI').absolute()
                datasets = os.listdir(path_datasets)
                ##########################################################
                for j in range(len(datasets)):
                    nameData = datasets[j][:len(datasets[j]) - 5]
                    if nameData in mean:
                        print(f"{nameData} already ran")
                        continue
                    # retrieve previous results
                    if os.path.exists(pathMean):
                        mean = pd.read_csv(pathMean, index_col=0)
                        timestamps = pd.read_csv(pathTimestamps, index_col=0)
                    start = str(datetime.now())
                    print(f"{nameData} started at {start}")

                    # runAlgo(1, nameData, settings)
                    try:
                        parallel_func = functools.partial(runAlgo, dataName = nameData, parameters = settings)

                        futures = client.map(parallel_func, ARGUMENTS)   

                        results = np.array(client.gather(futures))
                        end = str(datetime.now())
                        timestamps[nameData] = [start, end]
                        mean[nameData] = np.mean(results, axis=0)
                        mean.to_csv(f'{path_exp}/ARI')
                        timestamps.to_csv(f'{path_exp}/timestamps')
                        
                        print(f"{nameData} finished at {end}")
                    except Exception as e: # failed equals zero results
                        print(f"{nameData} failed with error")
                        print(e)
                        continue
                    

                    ##########################################################

                ##########################################################
                # when all datasets are ran, make the pictures and add to the bigger result
                print(f"Adding the results of {i}")
                path = Path(f"{path_exp}/pictures").absolute()
                CHECK_FOLDER = os.path.isdir(path)
                if not CHECK_FOLDER:
                    os.makedirs(path)
                    print("created folder : ", path)
                
                if os.path.exists(pathMean):
                    mean = pd.read_csv(pathMean, index_col=0)

                mean["total"] = mean.mean(axis=1) #axises founded on the internet
                all_results[i] = mean["total"]
                all_results.to_csv(path_results)

                mean.to_csv(f'{path_exp}/ARI')

                pic = pd.DataFrame()

                for key, value in mean.items():
                    pic["Cobras"] = cobras[key]
                    pic[i] = value

                    pic.plot()
                    plt.savefig(f"{path_exp}/pictures/{key}.png")

                # all_results[["Cobras", i]].plot()
                # plt.savefig(f"{path_exp}/total.png")

                ##########################################################

        # plot all the results
        all_results.plot()
        plt.savefig(f"{EXPERIMENT_PATH}/all.png")       

def test(nameData):
     with open(f'test.json') as json_file:
        experiments = json.load(json_file)

        # get the needed parameters
        experiment = experiments["cobrasparam"]

        # setup the arguments
        settings = dict() # is not a deepcopy if you have nested disctionaries (see documentation)
        if "metric" in experiment:
            metricsettings = copy.deepcopy(experiment["metric_parameters"])
            if "metric" in metricsettings:
                metricsettings["metric"]["value"] = eval(metricsettings["metric"]["value"])
            settings["metric"] = eval(experiment["metric"]) if experiment["metric"] else None
            settings["metric_parameters"] = metricsettings
        if "cluster_algo" in experiment:
            settings["cluster_algo"] = eval(experiment["cluster_algo"]) if experiment["cluster_algo"] else None
            settings["cluster_algo_parameters"] = experiment["cluster_algo_parameters"]

        if "rebuild_cluster" in experiment:
            settings["rebuild_cluster"] = eval(experiment["rebuild_cluster"]) if experiment["rebuild_cluster"] else None
            settings["rebuild_cluster_parameters"] = experiment["rebuild_cluster_parameters"]

        if "rebuilder" in experiment:
            settings["rebuilder"] = eval(experiment["rebuilder"]) if experiment["rebuilder"] else None
            settings["rebuilder_parameters"] = experiment["rebuilder_parameters"]

        if "baseline" in experiment:
            settings["baseline"] = experiment["baseline"]

        # plt.plot(np.arange(200),runAlgo(0, nameData, {}), label='cobras')
        plt.plot(np.arange(200),runAlgo(16, nameData, settings), label='test')
        plt.plot(np.arange(200),runAlgo(16, nameData, {}), label='COBRAS')

        # k = [5]
        
        # for i in k:
        #     settings["metric_parameters"]["queriesNeeded"] = i
        #     plt.plot(np.arange(200),runAlgo(1, nameData, settings), label=f'test_{i}')
        # plt.plot(np.arange(200),runAlgo(1, nameData, {}), label='COBRAS')
        
        plt.legend()
        plt.show()

def viz(nameData, queries, meticlearner):
    path = Path(f'datasets/cobras-paper/UCI/{nameData}.data').absolute()
    dataset = np.loadtxt(path, delimiter=',')
    data = dataset[:, 1:]
    target = dataset[:, 0]

    emb = TSNE(random_state=42).fit_transform(data)

    plt.scatter(x = emb[:,0], y =emb[:,1], c=target)
    plt.show()

    data = ITML_Supervised(random_state=42,num_constraints= 500).fit_transform(data, target)

    emb = TSNE(random_state=42).fit_transform(data)

    plt.scatter(x = emb[:,0], y =emb[:,1], c=target)
    plt.show()

def allvizs(seednumber:int):
    path = Path("Vizs").absolute()
    CHECK_FOLDER = os.path.isdir(path)
    if not CHECK_FOLDER:
        os.makedirs(path)
        print("created folder : ", path)



# def kNN(dataName):
#     path = Path(f'datasets/cobras-paper/UCI/{dataName}.data').absolute()
    
#     dataset = np.loadtxt(path, delimiter=',')
#     data = dataset[:, 1:]
#     target = dataset[:, 0]

#     ###############################""
#     querier = LabelQuerier(None, target, 100)
#     clusterer = COBRAS(correct_noise=False)
#     all_clusters, runtimes, superinstances, clusterIteration, transformations, ml, cl = clusterer.fit(data, -1, None, querier)
#     print(adjusted_rand_score(target, all_clusters[-1]))
#     pairs = np.vstack((ml,cl))
#     constrains = np.full(len(ml) + len(cl), 1)
#     constrains[len(ml):] = np.full(len(cl), -1)
#     ######################################

#     # model = KNeighborsClassifier(n_neighbors=3)
#     # model.fit(data,target)
#     # predicted = model.predict(data)

#     # print("data/target")
#     # print(adjusted_rand_score(target, predicted))

#     model = KNeighborsClassifier(n_neighbors=3)
#     model.fit(data,all_clusters[-1])
#     predicted = model.predict(data)

#     print("data/COBRAS")
#     print(adjusted_rand_score(target, predicted))

#     itml = ITML(preprocessor=data)
#     newdata = itml.fit(pairs, constrains).transform(data)

#     # model = KNeighborsClassifier(n_neighbors=3)
#     # model.fit(newdata,target)
#     # predicted = model.predict(newdata)

#     # print("newdata/target")
#     # print(adjusted_rand_score(target, predicted))

#     model = KNeighborsClassifier(n_neighbors=3)
#     model.fit(newdata, all_clusters[-1])
#     predicted = model.predict(newdata)

#     print("newdata/COBRAS")
#     print(adjusted_rand_score(target, predicted))
    
def executeExperiments():

    # ###################################
    # path = Path('experimenten/errors').absolute()
    # CHECK_FOLDER = os.path.isdir(path)
    # if not CHECK_FOLDER:
    #     os.makedirs(path)
    #     print("created folder : ", path)
    # ####################################

    #####################################
    path_data = Path('queue').absolute()
    dir_list = os.listdir(path_data)
    for k in dir_list:
        name = k
        print("Started "  + str(k))
        run(f'experimenten/{name}', f'queue/{name}')
        print("Finished "  + str(k))
        print("Deleting the queue of this experiment")

        shutil.rmtree(f'queue/{name}')
        ####################################
                

if __name__ == "__main__":
    def ignore_warnings():
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=Warning)

    ignore_warnings() # moet meegegeven worden met de workers tho
    # executeExperiments()
    # test("parkinsons")

                

