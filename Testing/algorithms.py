from abc import abstractmethod
from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.labelquerier import LabelQuerier
from noise_robust_cobras.metric_learning.metriclearning_algorithms import * #TODO: deze class herschrijven, handiger maken om subclasses
from metric_learn import NCA
from pathlib import Path
import os

# TODO, de jusite metric algo's meegeven

class Algorithm:
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def getDescription(self):
        pass

    @abstractmethod
    def getFileName(self):
        pass

    def preprocces(self, dataName, dataset_path):
        return dataset_path

##########
# Cobras #
##########
class Cobras(Algorithm):
    def __init__(self):
        pass

    def fit(self, dataName, path_for_data, maxQ, trainingset = None, prf = None):
        dataset = np.loadtxt(path_for_data, delimiter=',')
        data = dataset[:, 1:]
        target = dataset[:, 0]
        querier = LabelQuerier(None, target, maxQ, prf)
        clusterer = COBRAS(correct_noise=False)
        all_clusters, runtimes, *_ = clusterer.fit(data, -1, trainingset, querier)

        return all_clusters, runtimes

    def getDescription(self):
        return "COBRAS without any adaptions"

    def getFileName(self):
        return "COBRAS"

###########################################
# (semi)-Supervised after some iterations #
###########################################
class SemiSupervised(Algorithm):
    def __init__(self):
        pass

    def fit(self, dataName, path_for_data, maxQ, trainingset, prf = None):
        dataset = np.loadtxt(path_for_data, delimiter=',')
        data = dataset[:, 1:]
        target = dataset[:, 0]
        querier = LabelQuerier(None, target, maxQ, prf)
        clusterer = COBRAS(correct_noise=False, metric_algo = SemiSupervisedMetric())
        all_clusters, runtimes, *_ = clusterer.fit(data, -1, trainingset, querier)

        return all_clusters, runtimes

    def getDescription(self):
        return "COBRAS where after each iteration a metric is learned using all the constraints given so far"

    def getFileName(self):
        return "Semi_Supervised_ITML"

    def setMetric(self, ml):
        self.metric = SupervisedMetric() # dit moet nog gefixt worden

class Supervised(Algorithm):
    def __init__(self):
        pass

    def fit(self, dataName, path_for_data, maxQ, trainingset, prf = None):
        dataset = np.loadtxt(path_for_data, delimiter=',')
        data = dataset[:, 1:]
        target = dataset[:, 0]
        querier = LabelQuerier(None, target, maxQ, prf)
        clusterer = COBRAS(correct_noise=False, metric_algo = SupervisedMetric())
        all_clusters, runtimes, *_ = clusterer.fit(data, -1, trainingset, querier)

        return all_clusters, runtimes

    def getDescription(self):
        return "COBRAS where after each iteration a metric is learned using the clustering labels so far"

    def getFileName(self):
        return "Supervised_NCA"

    def setMetric(self, ml):
        self.metric = SupervisedMetric() # algo meegeven

#############################
# Preprocessing for testing #
#############################
class PortionPreprocessed(Algorithm):
    def __init__(self):
        pass

    def fit(self, dataName, path_for_data, maxQ, trainingset, prf = None): # logger nog uitbereiden
        dataset = np.loadtxt(path_for_data, delimiter=',')
        data = dataset[:, 1:]
        target = dataset[:, 0]
        pre = NCA(max_iter=100)
        pre.fit(np.copy(data[trainingset]), np.copy(target[trainingset]))
        newData = pre.transform(np.copy(data))
        querier = LabelQuerier(None, target, maxQ, prf)
        clusterer = COBRAS(correct_noise=False)
        all_clusters, runtimes, *_ = clusterer.fit(newData, -1, trainingset, querier)

        return all_clusters, runtimes

    def getDescription(self):
        return "COBRAS where a metric was first trained using the trainingset"

    def getFileName(self):
        return "COBRAS_training_preprocessed_NCA"

    def setMetric(self, ml):
        self.metric = SupervisedMetric() # algo meegeven

class Preprocessed(Algorithm): # maak een function preprocessed, want anders kan dit mislopen in de parallel setting (deze function wordt dan eerst opgeroepen)
    def __init__(self):
        pass

    def fit(self, dataName, path_for_data, maxQ, trainingset, prf = None): 
        dataset = np.loadtxt(path_for_data, delimiter=',')
        data = dataset[:, 1:]
        target = dataset[:, 0]
        querier = LabelQuerier(None, target, maxQ, prf)
        clusterer = COBRAS(correct_noise=False)    
        all_clusters, runtimes, *_ = clusterer.fit(data, -1, trainingset, querier)

        return all_clusters, runtimes
    
    def preprocces(self, dataName, dataset_path):
        path_pre = Path('batches/' + dataName + "_" + "preprocessed_NCA" ).absolute()    # + type(metricPreprocessing).__name__ voor later
        if os.path.exists(path_pre):
            return path_pre
        else:
            dataset = np.loadtxt(dataset_path, delimiter=',')
            data = dataset[:, 1:]
            target = dataset[:, 0]
            pre = NCA(max_iter=100)
            pre.fit(np.copy(data), np.copy(target))
            newData = pre.transform(np.copy(data))
            np.savetxt(path_pre, np.column_stack((target,newData)), delimiter=',')

            return path_pre

    def getDescription(self):
        return "COBRAS where a metric was first trained using all the data"

    def getFileName(self):
        return "COBRAS_preprocessed_NCA"

    def setMetric(self, ml):
        self.metric = SupervisedMetric() # algo meegeven

#############
# Baselines #
#############

class BaselineSemiSupervised(Algorithm):
    def __init__(self, algo = None): # ge gaat hier het algo moeten meegeven
        if algo:
            print("add it")
        self.metric = SupervisedMetric()
        pass
    
    # return a function because it is parallel
    def fit(self,dataName, path_for_data, maxQ, trainingset, prf = None): # need to first use via parameter in run function and then clear all and then use this, so this is a token function (only user this first if you do not want to reuse the previous learned metric which has been saved)
        dataset = np.loadtxt(path_for_data, delimiter=',')
        data = dataset[:, 1:]
        target = dataset[:, 0]
        querier = LabelQuerier(None, target, 100, prf)
        clusterer = COBRAS(correct_noise=False, metric_algo = SupervisedMetric(), end=True)
        all_clusters, runtimes, *_ = clusterer.fit(data, -1, trainingset, querier)

        querier2 = LabelQuerier(None, target, maxQ, prf)
        clusterer2 = COBRAS(correct_noise=False)
        all_clusters, runtimes, *_ = clusterer2.fit(clusterer.data, -1, trainingset, querier2)

        return all_clusters, runtimes

    def getDescription(self):
        return "COBRAS where a metric was first trained using all the data"

    def getFileName(self):
        return "Baseline_NCA"

    def setMetric(self, ml):
        self.metric = SupervisedMetric() # algo meegeven

