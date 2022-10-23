from abc import abstractmethod
from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.labelquerier import LabelQuerier
from noise_robust_cobras.metric_learning.metriclearning_algorithms import * #TODO: deze class herschrijven, handiger maken om subclasses
from metric_learn import NCA

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

##########
# Cobras #
##########
class Cobras(Algorithm):
    def __init__(self):
        pass

    def fit(self,data, target, maxQ, trainingset = None, prf = None):
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

    def fit(self, data, target, maxQ, trainingset = None, prf = None):
        querier = LabelQuerier(None, target, maxQ, prf)
        clusterer = COBRAS(correct_noise=False, metric_algo = SemiSupervisedMetric())
        all_clusters, runtimes, *_ = clusterer.fit(data, -1, trainingset, querier)

        return all_clusters, runtimes

    def getDescription(self):
        return "COBRAS where after each iteration a metric is learned using all the constraints given so far"

    def getFileName(self):
        return "Semi_Supervised_ITML"

class Supervised(Algorithm):
    def __init__(self):
        pass

    def fit(self, data, target, maxQ, trainingset = None, prf = None):
        querier = LabelQuerier(None, target, maxQ, prf)
        clusterer = COBRAS(correct_noise=False, metric_algo = SupervisedMetric())
        all_clusters, runtimes, *_ = clusterer.fit(data, -1, trainingset, querier)

        return all_clusters, runtimes

    def getDescription(self):
        return "COBRAS where after each iteration a metric is learned using the clustering labels so far"

    def getFileName(self):
        return "Supervised_NCA"

#############################
# Preprocessing for testing #
#############################
class PortionPreprocessed(Algorithm):
    def __init__(self):
        pass

    def fit(self, data, target, maxQ, trainingset = None, prf = None): # logger nog uitbereiden
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
        return "COBRAS_training_preprocessed"

class Preprocessed(Algorithm):
    def __init__(self):
        pass

    def fit(self, data, target, maxQ, trainingset = None, prf = None): # need to first use via parameter in run function and then clear all and then use this, so this is a token function (only user this first if you do not want to reuse the previous learned metric which has been saved)
        pre = NCA(max_iter=100)
        pre.fit(np.copy(data[trainingset]), np.copy(target[trainingset]))
        newData = pre.transform(np.copy(data))
        querier = LabelQuerier(None, target, maxQ, prf)
        clusterer = COBRAS(correct_noise=False)
        all_clusters, runtimes, *_ = clusterer.fit(newData, -1, trainingset, querier)

        return all_clusters, runtimes

    def getDescription(self):
        return "COBRAS where a metric was first trained using all the data"

    def getFileName(self):
        return "COBRAS_preprocessed"

#############
# Baselines #
#############

class BaselineSemiSupervised(Algorithm):
    def __init__(self, algo = None): # ge gaat hier het algo moeten meegeven
        if algo:
            print("add it")
        self.metric = SupervisedMetric()
        pass

    def fit(self, data, target, maxQ, trainingset = None, prf = None): # need to first use via parameter in run function and then clear all and then use this, so this is a token function (only user this first if you do not want to reuse the previous learned metric which has been saved)
        querier = LabelQuerier(None, target, maxQ, prf)
        clusterer = COBRAS(correct_noise=False, metric_algo = SupervisedMetric())
        all_clusters, runtimes, *_ = clusterer.fit(data, -1, trainingset, querier)

        return all_clusters, runtimes

    def getDescription(self):
        return "COBRAS where a metric was first trained using all the data"

    def getFileName(self):
        return "Baseline_NCA"

    def setMetric(self, ml):
        self.metric = SupervisedMetric() # algo meegeven

