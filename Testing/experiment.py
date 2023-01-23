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
import noise_robust_cobras.metric_learning.metriclearning_algorithms
import noise_robust_cobras.metric_learning.rebuildInstance
from noise_robust_cobras.metric_learning.metriclearning import *
# from metric_learn import * -> werken met wrappers

import numpy as np 
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
import sklearn as sk
from statistics import mean
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt

from dask.distributed import Client, LocalCluster
EXPERIMENT_PATH = 'experimenten/eerste'
nbRUNS = 100
ARGUMENTS = range(100)
SEED = 24
random_generator = np.random.default_rng(SEED)
seeds = [random_generator.integers(1,1000000) for i in range(nbRUNS)] # creqtion of the seeds
QUERYLIMIT = 200

def runAlgo(seed, dataName, parameters):
    path = Path(f'datasets/cobras-paper/UCI/{dataName}.data').absolute()
    dataset = np.loadtxt(path, delimiter=',')
    data = dataset[:, 1:]
    target = dataset[:, 0]
    querier = LabelQuerier(None, target, QUERYLIMIT)
    clusterer = COBRAS(correct_noise=False, seed=seeds[seed] **parameters)
    all_clusters, runtimes, *_ = clusterer.fit(data, -1, querier)
    if len(all_clusters) < QUERYLIMIT:
        diff = QUERYLIMIT - len(all_clusters)
        for ex in range(diff): all_clusters.append(all_clusters[-1])
    return [adjusted_rand_score(target, np.array(clustering)) for clustering in all_clusters]

def run():
    with LocalCluster() as cluster, Client(cluster) as client:

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
        path_cobras = Path(f"{EXPERIMENT_PATH}/COBRAS").absolute()
        if os.path.exists(pathMean):
            cobras = pd.read_csv(path_cobras, index_col=0)
        else:
            datasets = os.listdir(path_datasets)
            cobras = pd.DataFrame()
            for j in range(len(datasets)):
                nameData = dir_list[j][:len(dir_list[j]) - 5]
                print(f"Started running COBRAS")
                parallel_func = functools.partial(runAlgo.fit, dataName = nameData, parameters = {})
                futures = client.map(parallel_func, ARGUMENTS)   
                results = np.array(client.gather(futures))
                cobras[nameData] = np.mean(results, axis=1)
                print(f"Finished running COBRAS")
            cobras["total"] = cobras.mean(axis=0)
            all_results["Cobras"] = cobras["total"]
            all_results.to_csv(path_results)
            cobras.to_csv(f"{EXPERIMENT_PATH}/COBRAS")
        ##########################################################


        ##########################################################
        # gaan ervan uit dat alles in de queue van een experiment is
        path = Path(EXPERIMENT_PATH).absolute()
        CHECK_FOLDER = os.path.isdir(path)
        if not CHECK_FOLDER:
            os.makedirs(path)
            print("created folder : ", path)
        ##########################################################


        ##########################################################
        path_data = Path('queue').absolute()
        dir_list = os.listdir(path_data)
        for i in dir_list:
            with open(f'queue/{i}') as json_file:
                print(f"Start/continue {i}")
                # ga verder met een experiment of start
                experiments = json.load(json_file)

                experiment = experiments["cobrasparam"]

                # setup the arguments
                settings = dict()
                metricsettings = dict(settings["metric_parameters"])
                if "metric" in metricsettings:
                    metricsettings["metric"]["value"] = eval(metricsettings["metric"]["value"])
                settings["metric"] = eval(experiment["metric"])(metricsettings)
                if "cluster_algo" in settings:
                    settings["cluster_algo"] = eval(experiment["cluster_algo"])(experiment["cluster_algo_parameters"])
                if "rebuild_cluster" in settings:
                    settings["rebuild_cluster"] = eval(experiment["rebuild_cluster"])(experiment["rebuild_cluster_parameters"])
                if "rebuilder" in settings:
                    settings["rebuilder"] = eval(experiment["rebuilder"])(experiment["rebuilder_parameters"])
                
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

                with open(f"{path_exp}/settings.json", "w") as outfile:
                    json.dump(experiment, outfile, indent=4)
                
                # go over all the UCI datasets
                path_datasets = Path('datasets/cobras-paper/UCI').absolute()
                datasets = os.listdir(path_datasets)
                ##########################################################
                for j in range(len(datasets)):
                    nameData = dir_list[j][:len(dir_list[j]) - 5]
                    if nameData in mean:
                        print(f"{nameData} already ran")
                        continue
                    # retrieve previous results
                    if os.path.exists(pathMean):
                        mean = pd.read_csv(pathMean, index_col=0)
                        timestamps = pd.read_csv(pathTimestamps, index_col=0)
                    start = str(datetime.now())
                    print(f"{nameData} started at {start}")
                    parallel_func = functools.partial(runAlgo.fit, dataName = nameData, parameters = settings)

                    futures = client.map(parallel_func, ARGUMENTS)   

                    results = np.array(client.gather(futures))
                    end = str(datetime.now())
                    timestamps[nameData] = [start, end]
                    mean[nameData] = np.mean(results, axis=1)
                    mean.to_csv(f'{path_exp}/ARI')
                    timestamps.to_csv(f'{path_exp}/timestamps')
                    
                    print(f"{nameData} finished at {end}")
                    

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

                mean["total"] = mean.mean(axis=0)
                all_results[i] = mean["total"]
                all_results.to_csv(path_results)

                mean.to_csv(f'{path_exp}/ARI')

                pic = pd.DataFrame()

                for key, value in mean.items():
                    totalMean += value
                    pic["Cobras"] = cobras[key]
                    pic[i] = value

                    pic.plot()
                    plt.savefig(f"{path_exp}/{i}.png")

                all_results[["Cobras", i]].plot()
                plt.savefig(f"{path_exp}/total.png")

                ##########################################################

                

