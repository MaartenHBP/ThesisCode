from metric_learn import *
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import SpectralClustering
from abc import abstractmethod
import numpy as np
from scipy.sparse import csr_matrix

class MetricLearner:
    def __init__(self, preprocessor = None):
        self.affinity = None
        self.preprocessor = preprocessor
    @abstractmethod
    def fit(self, pairs ,y, local = None): # dingen die tansformen en dingen die affinity maken kunnen gecombineerd al worden (bv spectral na ITML (the full power))
        pass # return transformed and affinity
    @abstractmethod
    def transform(self, data):
        pass

class ITML_wrapper(MetricLearner):
    def __init__(self, preprocessor=None):
        self.fitted = None
        super().__init__(preprocessor) # TODO: dit uitbereiden

    def fit(self, pairs, y, local = None):
        self.fitted = ITML(preprocessor=self.preprocessor)
        self.fitted.fit(pairs, y)
        return self

    def transform(self, data):
        return self.fitted.transform(data), self.affinity

class NCA_wrapper(MetricLearner):
    def __init__(self, preprocessor=None):
        self.fitted = None
        super().__init__(preprocessor) # TODO: dit uitbereiden

    def fit(self, pairs, y, local = None):
        self.fitted = NCA()
        self.fitted.fit(pairs, y)
        return self

    def transform(self, data):
        return self.fitted.transform(data), self.affinity

class Spectral(MetricLearner):
    def __init__(self, preprocessor=None):
        super().__init__(preprocessor)
    
    def fit(self, pairs, y, local = None):
        data = self.preprocessor if local is None else self.preprocessor[local,:]
        n_neighbours = min(10, len(data))
        sp = SpectralClustering(n_clusters=1, eigen_solver="arpack", affinity='nearest_neighbors', n_neighbors=n_neighbours).fit(data)
        aff = csr_matrix((len(self.preprocessor), len(self.preprocessor)))
        if local is not None:
            aff[local, :][:,local] = sp.affinity_matrix_
        else:
            aff = sp.affinity_matrix_
        yc = np.copy(y)
        yc[yc==-1] = 0
        aff[pairs[:,0], pairs[:,1]] = yc
        aff[pairs[:,1], pairs[:,0]] = yc
        # self.affinity = SpectralEmbedding(eigen_solver="arpack", affinity='precomputed').fit_transform(aff)
        self.affinity = aff
        return self

    def transform(self, data):
        return np.copy(data), self.affinity # TODO:data kan ook getransformeerd worden, dus mss best nog doen
