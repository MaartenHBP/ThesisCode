import numpy as np
from pathlib import Path
from abc import abstractmethod
from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.labelquerier import LabelQuerier
from noise_robust_cobras.metric_learning.metriclearning_algorithms import * 
from metric_learn import *
from pathlib import Path
import os
import pandas as pd

class metricSettings:
    def __init__(self, metric, data, parameters, name, typeMetric):
        self.name = name
        self.typeMetric = typeMetric
        self.metric = metric

        # path = Path(f'{datasetpath}.data').absolute()
        # data= np.loadtxt(path, delimiter=',')
        # self.dataset = data

        self.data = data

        self.parameters = parameters
        self.constraints = None
        self.transformed = None
        self.cobrasOriginal = None   
        self.cobrasTransformed = None # should be animations

    def changeMertic(self, metric, parameters, typeMetric):
        self.metric = metric
        self.typeMetric = typeMetric
        self.parameters = parameters
        self.transformed = None
        self.cobrasTransformed = None

    def changeData(self, data):
        self.data = data
        self.transformed = None
        self.cobrasTransformed = None
        self.cobrasOriginal = None

    def changeName(self, name):
        self.name = name
        return self

    def copy(self, name):
        return metricSettings(self.metric, self.data, dict(self.parameters), name, self.typeMetric)

    def newConstraints(self):
        if (self.typeMetric == "supervised"):
            print("supervised constraints")

        else:
            print("semisupervised constraints")

    def learnMetric(self, onOrig = True):
        if onOrig:
            print("learning metric on original data")
        else:
            print("learning metric on transformed data")
        self.transformed = None
        self.cobrasTransformed = None

    def executeCOBRAS(self, data, onOrig = True):
        if self.transformed and not onOrig:
            labels = self.transformed[:,0]
            querier2 = LabelQuerier(None, labels, 200)
            clusterer = COBRAS(correct_noise=False, logExtraInfo=True)
            all_clusters, runtimes, superinstances, clusterIteration, *_ = clusterer.fit(self.transformed[:,1:], -1, None, querier2)
            self.cobrasTransformed = [superinstances, clusterIteration]

        if onOrig:
            labels = data[:,0]
            querier2 = LabelQuerier(None, labels, 200)
            clusterer = COBRAS(correct_noise=False, logExtraInfo=True)
            all_clusters, runtimes, superinstances, clusterIteration, *_ = clusterer.fit(data[:,1:], -1, None, querier2)
            self.cobrasOriginal = [superinstances, clusterIteration]

        # animation should be saved here

