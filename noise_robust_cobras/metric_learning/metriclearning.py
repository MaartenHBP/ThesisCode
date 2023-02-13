from metric_learn import *
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import SpectralClustering
from abc import abstractmethod
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize

from numpy.linalg import matrix_rank
from numpy import linalg as LA
from scipy.linalg import eigh
from sklearn.metrics import pairwise_distances
import math

from sklearn.manifold._locally_linear import barycenter_kneighbors_graph

from sklearn.metrics import pairwise_kernels # can be very usefull
from scipy.spatial.distance import cdist



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


class kernelbased(MetricLearner):
    ''' 
    Semi-Supervised Metric Learning Using Pairwise Constraints, Soleymani Baghshah M, Bagheri Shouraki S
    '''
    def __init__(self, preprocessor = None):
        self.ensemble = None
        self.distance = pairwise_distances(preprocessor, metric='Euclidian')
        self.d = math.floor(preprocessor.shape[1]/2)
        # self.d = 2
        self.alph = 0.2 if self.d > 5 else 0.02
        self.k = 10
        self.prepocessor = preprocessor

    def fit(self, pairs ,y, local = None):
        n = len(self.distance)
        w = 2*1.5*np.sum(np.triu(self.distance))/(n*(n + 1)) # hier zijn er ook parameters waarmee je kan spelen
        kernel = np.exp(self.distance/-w)
        seen_indices = set()

        ML = pairs[y == 1]
        CL = pairs[y == -1]
        for ml in ML:
            seen_indices.add(ml[0])
            seen_indices.add(ml[1])
        ml_indices = list(seen_indices)
        reduced_kernel = kernel[ml_indices, :] # ik denk dat het zo moet

        # use the constraints
        Sp = np.zeros((n, n))
        Sp[ML[:,0], ML[:,1]] = 1
        Sp[ML[:,1], ML[:,0]] = 1
        Dp = np.diag(Sp.sum(axis=1))
        Up = Dp - Sp

        Sd = np.zeros((n, n))
        Sd[CL[:,0], CL[:,1]] = 1
        Sd[CL[:,1], CL[:,0]] = 1
        Dd = np.diag(Sd.sum(axis=1))
        Ud = Dd - Sd

        Sb = reduced_kernel @ Ud @ reduced_kernel.T
        
        optimal_weight_matrix = barycenter_kneighbors_graph(self.prepocessor, n_neighbors=self.k)

        E = (np.identity(n) - optimal_weight_matrix).T @ (np.identity(n) - optimal_weight_matrix)
        Sw = reduced_kernel @ (Up + self.alph * E) @ reduced_kernel.T

        self.ensemble = reduced_kernel.T @ heursitcSearch(Sw, Sb, self.d, 0.00000000001)

        return self

    def transform(self, data):
        return self.ensemble, self.affinity

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
    
class RCA_wrapper(MetricLearner):
    def __init__(self, preprocessor = None, n_components = None):
        self.ensemble = None
        self.preprocessor = preprocessor
    def fit(self, pairs ,y):
        blobs = []
        seen_indices = [] # deze zitten dus in ML blobs
        for ml in pairs[y == 1]:
            ind1 = ml[0]
            ind2 = ml[1]
            blob1 = []
            blob2 = []
            if ind1 in seen_indices:
                for blob in blobs:
                    if ind1 in blob:
                        blob1 = blob
                        break
            if ind2 in seen_indices:
                for blob in blobs:
                    if ind2 in blob:
                        blob2 = blob
                        break

            if len(blob1) > 0 and len(blob2) > 0:
                blob1.extend(blob2)
                blobs.remove(blob2)
                continue
            if len(blob1) > 0:
                blob1.append(ind2)
                seen_indices.append(ind2)
                continue
            if len(blob2) > 0:
                blob2.append(ind1)
                seen_indices.append(ind1)
                continue
            blobs.append([ind1, ind2])
            seen_indices.extend([ind1, ind2])
        constr = []
        indici = []
        i = 0
        for blob in blobs:
            indici.extend(blob)
            constr.extend([i]*len(blob))
            i+=1
        self.ensemble = RCA(preprocessor=self.preprocessor).fit(indici, constr)
        return self

    def transform(self, data):
        return self.ensemble.transform(data), self.affinity

class Spectral(MetricLearner): # TODO: dit opnieuw testen, wrs is hier een fout
    def __init__(self, preprocessor=None):
        super().__init__(preprocessor)
    
    def fit(self, pairs, y, local = None):
        data = self.preprocessor if local is None else self.preprocessor[local,:]
        n_neighbours = min(10, len(data))
        sp = SpectralClustering(n_clusters=1, eigen_solver="arpack", affinity='nearest_neighbors', n_neighbors=n_neighbours).fit(data)
        aff = csr_matrix((len(self.preprocessor), len(self.preprocessor)))
        if local is not None:
            aff[np.ix_(local, local)]  = sp.affinity_matrix_ # this is correct
        else:
            aff = sp.affinity_matrix_
        yc = np.copy(y)
        yc[yc==-1] = 0
        aff[pairs[:,0], pairs[:,1]] = yc
        aff[pairs[:,1], pairs[:,0]] = yc
        # self.affinity = SpectralEmbedding(eigen_solver="arpack", affinity='precomputed').fit_transform(aff) -> ook nog zo transformeren
        self.affinity = aff
        return self

    def transform(self, data):
        n_components = math.floor(data.shape[1]/2)# aantal components is nbatuurlijk wel belangrijk
        transformed = SpectralEmbedding(eigen_solver="arpack", affinity='precomputed', n_components= n_components).fit_transform(self.affinity)
        return transformed, self.affinity
    

####################
# Helper functions #
####################
def heursitcSearch(Sw, Sb, d, errorterm):
    # make the new Sw and Sb -> doet kinda iets juist ma is nog niet echt duidelijk hoe ik dit moet incorporeren
    # A = Sw + Sb
    # w, v = eigh(A)
    # W1 = v[w != 0]
    # Sw = W1.T @ Sw @ W1
    # Sb = W1.T @ Sb @ W1

    rank = matrix_rank(Sw)
    if d > len(Sw) - rank:
        lambda1 = np.trace(Sb)/np.trace(Sw)
        Sb_eigenvalues = eigh(Sb, eigvals_only=True)[-d:]
        alphas = np.sum(Sb_eigenvalues)
        Sw_eigenvalues = eigh(Sw, eigvals_only=True)[-d:]
        betas = np.sum(Sw_eigenvalues)
        lambda2 = alphas/betas
        lmbda = (lambda1 + lambda2)/2

        while lambda2 - lambda1 > errorterm:
            print("entered")
            g = np.sum(eigh(Sb - lmbda*Sw , eigvals_only=True)[-d:])
            if g > 0: lambda1 = lmbda
            else: lambda2 = lmbda
            lmbda = (lambda1 + lambda2)/2
        _, v = eigh(Sb - lmbda*Sw)
        W = v[:,-d:]
        return W
        # return W1 @ W @ W.T @ W1.T

    else:
        eigen, z = eigh(Sw)
        Z = z[:,0:(len(Sw) - rank)]
        _, v = eigh(Z.T@Sb@Z)
        W = v[:,-d:]
        print((Z@W).shape)
        return Z @ W
        # return W1 @ W @ W.T @ W1.T
