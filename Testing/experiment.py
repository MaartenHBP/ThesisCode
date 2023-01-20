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
from noise_robust_cobras.metric_learning.metricLearners import *
from metric_learn import *

import numpy as np 
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
import sklearn as sk
from statistics import mean
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt

from dask.distributed import Client, LocalCluster
EXPERIMENT_PATH = 'experimenten/eerste'

def runAlgo():
    pass

def run():
    with LocalCluster() as cluster, Client(cluster) as client:
        # gaan ervan uit dat alles in de queue van een experiment is
        path = Path(EXPERIMENT_PATH).absolute()
        CHECK_FOLDER = os.path.isdir(path)
        if not CHECK_FOLDER:
            os.makedirs(path)
            print("created folder : ", path)
        path_data = Path('queue').absolute()
        dir_list = os.listdir(path_data)
        for i in dir_list:
            with open(f'queue/{i}') as json_file:
                print(f"Start/continue {i}")
                # ga verder met een experiment of start
                experiment = json.load(json_file)

                # setup the arguments

                path_exp = f'{EXPERIMENT_PATH}/{i}'

                path = Path(path_exp).absolute()
                CHECK_FOLDER = os.path.isdir(path)
                if not CHECK_FOLDER:
                    os.makedirs(path)
                    print("created folder : ", path)

                pathMean = Path(f'{path_exp}/ARI')
                pathTimestamps = Path(f'{path_exp}/ARI')
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
                for i in range(len(datasets)):
                    nameData = dir_list[i][:len(dir_list[i]) - 5]
                    if nameData in mean:
                        continue

                    # retrieve previous results
                    if os.path.exists(pathMean):
                        mean = pd.read_csv(pathMean, index_col=0)
                        timestamps = pd.read_csv(pathTimestamps, index_col=0)

                    # execute the algorith


