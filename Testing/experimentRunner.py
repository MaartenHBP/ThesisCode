from array import array
from audioop import mul
from math import sqrt
import math
from pathlib import Path
import os
from statistics import mean
from sklearn.metrics import adjusted_rand_score
from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.noisy_labelquerier import ProbabilisticNoisyQuerier
from noise_robust_cobras.querier.labelquerier import LabelQuerier
import noise_robust_cobras.metric_learning.metriclearning_algorithms 
import numpy as np 
import scipy
from metric_learn import NCA
from batch import Batch
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
import sklearn as sk
from experimentLogger import ExperimentLogger
import matplotlib.pyplot as plt
import shutil
import functools
from dask.distributed import Client, LocalCluster
from test_scripts import *
from algorithms import *
from pathlib import Path
from metric_learn import *
from operator import itemgetter
import json

def fixClass(dictio):
    if "type" in dictio.keys():
        if dictio["type"] == "class":
            dictio["value"] = eval(dictio["value"])
    for key,value in dictio.items():
        if type(value) is dict:
            fixClass(value)

def run():
    with LocalCluster() as cluster, Client(cluster) as client:
        path_data = Path('queue').absolute()
        dir_list = os.listdir(path_data)
        for i in dir_list: # execute each file
            with open(f'queue/{i}') as json_file:
                experiment = json.load(json_file)
                fixClass(experiment)
                foldnb = experiment["foldnb"]
                crossfold = experiment["crossfold"]
                data = experiment["data"]
                runsPQ = experiment["runsPQ"]
                settings = experiment["settings"]
                name = experiment["string"]
                print(f"==={name}===")
                results = {}
                pathS1 = Path(f'batches/ARI/{name}')
                pathS2 = Path(f'batches/S2/{name}')
                pathTime = Path(f'batches/Time/{name}')
                S1 = pd.DataFrame()
                S2 = pd.DataFrame()
                time = pd.DataFrame()
                if os.path.exists(pathS1):
                    S1 = pd.read_csv(pathS1, index_col=0)
                    S2 = pd.read_csv(pathS2, index_col=0)
                    time = pd.read_csv(pathTime, index_col=0)
                # preprocess the data
                print("preprocessing")
                parallel_func = functools.partial(Algorithm.preprocess, **settings["fitparam"])
                futures = client.map(parallel_func, data)
                results = client.gather(futures)
                for nameData in data: # voor niet crossfold moet ge gewoon de argumenten anders opbouwen
                    if nameData in S1:
                        continue
                    dataset_path = Path('datasets/cobras-paper/' + nameData + '.data').absolute()
                    dataset = np.loadtxt(dataset_path, delimiter=',')
                    target = dataset[:, 0]
                    average = {"S1": np.zeros(200), "S2": np.zeros(200), "times": np.zeros(200)}
                    resulting_path = Path(f'batches/folds/{nameData}_crossfold_{foldnb}').absolute()
                    folds = np.loadtxt(resulting_path, delimiter=',', dtype=int)

                    arguments = []
                    test = []

                    size = len(folds[0])
                    amount = math.ceil(0.1 * size)

                    for fold in folds:
                        arguments.append(fold[0:-amount])
                        test.append(fold[-amount:])

                    print(data)
                    parallel_func = functools.partial(Algorithm.fit, dataName = nameData, **settings["fitparam"], parameters = settings["cobrasparam"])

                    futures = client.map(parallel_func, arguments)   

                    results = client.gather(futures)

                    print("Done, total results = " + str(len(results)), end = "\r")

                    for res in range(len(results)):
                        all_clusters, runtimes = results[res]
                        if len(all_clusters) < 200:
                            diff = 200 - len(all_clusters)
                            for ex in range(diff):
                                all_clusters.append(all_clusters[-1])
                                runtimes.append(runtimes[-1])
                            
                        IRA = np.array([adjusted_rand_score(target[test[res]], np.array(clustering)[test[res]]) for clustering in all_clusters])
                        average["S1"] += IRA
                        average["S2"] += IRA**2
                        average["times"] += np.array(runtimes)
                    S1[nameData] = average["S1"]
                    S2[nameData] = average["S2"]
                    time[nameData] = average["times"]
                S1.to_csv(f'batches/ARI/{name}')
                S2.to_csv(f'batches/S2/{name}')
                time.to_csv(f'batches/Time/{name}')
            os.remove(Path(f'queue/{i}').absolute())    

if __name__ == "__main__":
    def ignore_warnings():
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=Warning)

    ignore_warnings() # moet meegegeven worden met de workers tho
    run()