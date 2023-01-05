from abc import abstractmethod
from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.labelquerier import LabelQuerier
from noise_robust_cobras.metric_learning.metriclearning_algorithms import * 
from metric_learn import *
from pathlib import Path
import os
import pandas as pd

class Algorithm:

    @staticmethod
    def fit(trainingset, dataName, preprocessor = None, preprocestraining = False, baseline = False, parameters = {}):
        path = Path(f'datasets/cobras-paper/UCI/{dataName}.data').absolute() # ff UCI gehardcode
        if preprocessor and not preprocestraining: 
            paramstr = str(preprocessor["parameters"].values()).replace(" ", "").replace("[", "").replace("]","").replace(",", "_").replace("'","")
            path = Path(f'batches/preprocessing/{dataName}_{preprocessor["value"].__name__}_{paramstr}').absolute()
        dataset = np.loadtxt(path, delimiter=',')
        data = dataset[:, 1:]
        target = dataset[:, 0]
        if preprocessor and preprocestraining:
            pre = preprocessor["value"](**preprocessor["parameters"])
            pre.fit(np.copy(data[trainingset]), np.copy(target[trainingset]))
            data = pre.transform(np.copy(data))
        querier = LabelQuerier(None, target, 200)
        clusterer = COBRAS(correct_noise=False, end=baseline, **parameters)
        all_clusters, runtimes, *_ = clusterer.fit(data, -1, trainingset, querier)

        if baseline:
            querier2 = LabelQuerier(None, target, 200)
            clusterer2 = COBRAS(correct_noise=False, **parameters)
            all_clusters, runtimes, *_ = clusterer2.fit(clusterer.data, -1, trainingset, querier2)

        return all_clusters, runtimes

    @staticmethod
    def preprocess(dataName, preprocessor = None, preprocestraining = False, **args): # nog fixen, gaat ook parallel worden uitgevoerd
        if preprocestraining or not preprocessor:
            return
        paramstr = str(preprocessor["parameters"].values()).replace(" ", "").replace("[", "").replace("]","").replace(",", "_").replace("'","")
        path = Path(f'datasets/cobras-paper/UCI/{dataName}.data').absolute()
        path_pre = Path(f'batches/preprocessing/{dataName}_{preprocessor["value"].__name__}_{paramstr}').absolute()    # + type(metricPreprocessing).__name__ voor later
        if os.path.exists(path_pre):
            return
        else:
            dataset = np.loadtxt(path, delimiter=',')
            data = dataset[:, 1:]
            target = dataset[:, 0]
            pre = preprocessor["value"](**preprocessor["parameters"])
            pre.fit(np.copy(data), np.copy(target))
            newData = pre.transform(np.copy(data))
            np.savetxt(path_pre, np.column_stack((target,newData)), delimiter=',')

