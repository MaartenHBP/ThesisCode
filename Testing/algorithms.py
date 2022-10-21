from abc import abstractmethod
from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.labelquerier import LabelQuerier
from noise_robust_cobras.metric_learning.metriclearning_algorithms import * #TODO: deze class herschrijven, handiger maken om subclasses

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

# plain old COBRAS
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

