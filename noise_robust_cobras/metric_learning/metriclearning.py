from metric_learn import *
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import SpectralClustering
from abc import abstractmethod
import numpy as np

class MetricLearner:
    def __init__(self, preprocessor = None):
        self.affinity = None
        self.preprocessor = preprocessor
    @abstractmethod
    def fit(self, pairs ,y): # dingen die tansformen en dingen die affinity maken kunnen gecombineerd al worden (bv spectral na ITML (the full power))
        pass # return transformed and affinity
    @abstractmethod
    def transform(self, data):
        pass

class ITML_wrapper(MetricLearner):
    def __init__(self, preprocessor=None):
        self.fitted = None
        super().__init__(preprocessor) # TODO: dit uitbereiden

    def fit(self, pairs, y):
        self.fitted = ITML(preprocessor=self.preprocessor)
        self.fitted.fit(pairs, y)
        return self

    def transform(self, data):
        return self.fitted.transform(data), self.affinity

class Spectral(MetricLearner):
    def __init__(self, preprocessor=None):
        super().__init__(preprocessor)
    
    def fit(self, pairs, y):
        sp = SpectralClustering(n_clusters=2, eigen_solver="arpack", affinity='nearest_neighbors').fit(self.preprocessor)
        aff = sp.affinity_matrix_
        yc = np.copy(y)
        yc[yc==-1] = 0
        aff[pairs[:,0], pairs[:,1]] = yc
        aff[pairs[:,1], pairs[:,0]] = yc
        self.affinity = SpectralEmbedding(eigen_solver="arpack", affinity='precomputed').fit_transform(aff)
        return self

    def transform(self, data):
        return np.copy(data), self.affinity
