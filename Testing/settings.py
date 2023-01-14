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
from sklearn.cluster import KMeans
import math
from sklearn.cluster import SpectralClustering

class metricSettings:
    def __init__(self, metric, data, parameters, name, typeMetric, k = 2):
        self.name = name
        self.typeMetric = typeMetric
        self.metric = metric

        # path = Path(f'{datasetpath}.data').absolute()
        # data= np.loadtxt(path, delimiter=',')
        # self.dataset = data

        self.data = data

        self.parameters = parameters
        self.constraints = None
        self.pairs = None
        self.transformed = None
        self.clustOriginal = None   
        self.clustTransformed = None
        self.k = k

        self.clustering = None

    def changeMertic(self, metric, parameters, typeMetric):
        self.metric = metric
        self.typeMetric = typeMetric
        self.parameters = parameters
        self.transformed = None
        self.clustOriginal = None   
        self.clustTransformed = None

    def changeData(self, data):
        self.data = data
        self.transformed = None
        self.clustOriginal = None   
        self.clustTransformed = None

    def changeName(self, name):
        self.name = name
        return self

    def changeK(self, newK):
        self.k = newK

    def copy(self, name):
        new = metricSettings(self.metric, self.data, dict(self.parameters), name, self.typeMetric, self.k)
        new.constraints = np.copy(self.constraints) # zeer nuttig om testen uit te voeren
        new.pairs = np.copy(self.pairs)
        return new

    def newConstraints(self, data, nb_links):
        if (self.typeMetric == "supervised"):
            print("supervised constraints")
            self.constraints = np.copy(data[:,0])

        else:
            AMOUNT_OF_LINKS_WANTED = nb_links # also make this a parameter
            print("semisupervised constraints")
            labels = data[:,0] # mss hier ook nog met transformed werken
            nbLinks = labels.shape[0]
            maxQ = math.floor(AMOUNT_OF_LINKS_WANTED)
            querier = LabelQuerier(None, labels, maxQ)
            clusterer = COBRAS(correct_noise=False)
            all_clusters, runtimes, superinstances, clusterIteration, transformations, ml, cl = clusterer.fit(data[:,1:], -1, None, querier)
            pairs = np.vstack((ml,cl))
            constrains = np.full(len(ml) + len(cl), 1)
            constrains[len(ml):] = np.full(len(cl), -1)
            self.constraints = constrains
            self.pairs = pairs

    def learnMetric(self, data, onOrig = True):
        if self.constraints is None:
            print("Need constraints")
            return
        if onOrig:
            print("learning metric on original data")
            if (self.typeMetric == "supervised"):
                metric = eval(self.metric)(**self.parameters)
                self.transformed = metric.fit(np.copy(data), self.constraints).transform(np.copy(data))
            else:
                metric = eval(self.metric)(preprocessor=np.copy(data),**self.parameters)
                self.transformed = metric.fit(self.pairs, self.constraints).transform(np.copy(data))
            
        else:
            if not self.transformed is None:
                print("learning metric on transformed data")
                if (self.typeMetric == "supervised"):
                    metric = eval(self.metric)(**self.parameters)
                    self.transformed = metric.fit(np.copy(self.transformed), self.constraints).transform(np.copy(self.transformed))
                else:
                    metric = eval(self.metric)(preprocessor=np.copy(data),**self.parameters)
                    self.transformed = metric.fit(self.pairs, self.constraints).transform(np.copy(self.transformed))

            else:
                print("No transformed data available")
            
        self.clustOriginal = None   
        self.clustTransformed = None

    def executeClustering(self, data):
        kmeans = KMeans(self.k).fit(data)
        # kmeans = SpectralClustering(n_clusters=2,
        # eigen_solver="arpack", affinity='nearest_neighbors').fit(data)
        # kmeans.fit(data)
        self.clustOriginal = kmeans.labels_
        if not self.transformed is None:
            kmeans = KMeans(self.k)
            kmeans.fit(self.transformed)
            self.clustTransformed = kmeans.labels_


    # def executeCOBRAS(self, data, onOrig = True):
    #     if self.transformed and not onOrig:
    #         labels = self.transformed[:,0]
    #         querier2 = LabelQuerier(None, labels, 200)
    #         clusterer = COBRAS(correct_noise=False, logExtraInfo=True)
    #         all_clusters, runtimes, superinstances, clusterIteration, *_ = clusterer.fit(self.transformed[:,1:], -1, None, querier2)
    #         self.cobrasTransformed = [superinstances, clusterIteration]

    #     if onOrig:
    #         labels = data[:,0]
    #         querier2 = LabelQuerier(None, labels, 200)
    #         clusterer = COBRAS(correct_noise=False, logExtraInfo=True)
    #         all_clusters, runtimes, superinstances, clusterIteration, *_ = clusterer.fit(data[:,1:], -1, None, querier2)
    #         self.cobrasOriginal = [superinstances, clusterIteration]

        # animation should be saved here

