from metric_learn import *
from dml import KLMNN


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

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

import itertools


class MetricLearner:
    def __init__(self, preprocessor = None, expand = False):
        self.preprocessor = preprocessor
        self.expand = expand
    @abstractmethod
    def fit(self, pairs , y, points, labels): # points en labels is fully labelled data
        pass # return transformed and affinity
    @abstractmethod
    def transform(self, data):
        pass


class kernelbased(MetricLearner): # TODO: check de implementatie
    ''' 
    Semi-Supervised Metric Learning Using Pairwise Constraints, Soleymani Baghshah M, Bagheri Shouraki S
    '''
    def __init__(self, preprocessor = None):
        self.ensemble = None
        self.distance = pairwise_distances(preprocessor, metric='euclidean')
        self.d = math.floor(preprocessor.shape[1]/2)
        # self.d = 2
        self.alph = 0.2 if self.d > 5 else 0.02
        self.k = 10
        super().__init__(preprocessor)

    def fit(self, pairs ,y, points, labels):
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
        
        optimal_weight_matrix = barycenter_kneighbors_graph(self.preprocessor, n_neighbors=self.k)

        E = (np.identity(n) - optimal_weight_matrix).T @ (np.identity(n) - optimal_weight_matrix)
        Sw = reduced_kernel @ (Up + self.alph * E) @ reduced_kernel.T

        self.ensemble = reduced_kernel.T @ heursitcSearch(Sw, Sb, self.d, 0.00000000001)

        return self

    def transform(self, data):
        return self.ensemble, self.affinity
    
class ITMLNCA(MetricLearner):
    def __init__(self, preprocessor=None, expand = False, seed = 42):
        self.fitted = None
        self.seed = seed
        super().__init__(preprocessor, expand) # TODO: dit uitbereiden

    def fit(self, pairs, y, points, labels):
        if self.expand:
            pairs, y = expand(pairs, y)
        supervised = NCA(random_state= self.seed)
        supervised.fit(self.preprocessor[np.array(points)], labels)
        self.fitted = ITML(random_state=self.seed, preprocessor=self.preprocessor, max_iter=100, prior=supervised.get_mahalanobis_matrix())
        self.fitted.fit(pairs, y)
        return self

    def transform(self, data):
        return self.fitted.transform(data)
    
    def fit_transform(self, pairs, y, points, labels):
        self.fit(pairs, y, points, labels)
        return self.transform(self.preprocessor)

class ITML_wrapper(MetricLearner):
    def __init__(self, preprocessor=None, expand = False, seed = 42):
        self.fitted = None
        self.seed = seed
        super().__init__(preprocessor, expand) # TODO: dit uitbereiden

    def fit(self, pairs, y, points, labels):
        if self.expand:
            pairs, y = expand(pairs, y)
        self.fitted = ITML(random_state=self.seed, preprocessor=self.preprocessor, max_iter=100)
        self.fitted.fit(pairs, y)
        return self

    def transform(self, data):
        return self.fitted.transform(np.copy(data))
    
    def fit_transform(self, pairs, y, points, labels):
        self.fit(pairs, y, points, labels)
        return self.transform(np.copy(self.preprocessor))

class NCA_wrapper(MetricLearner):
    def __init__(self, preprocessor=None, seed = 42):
        self.fitted = None
        self.seed = seed
        super().__init__(preprocessor) # TODO: dit uitbereiden

    def fit(self, pairs, y, points, labels):
        # als er labels minder dan 3 keer voorkomen, eruit halen ja
        unique, counts = np.unique(labels, return_counts=True)
        problem = unique[counts < 3]
        newpoint, newlabels = points, labels
        if (len(problem) > 0):
            select = np.invert(np.in1d(labels, problem))
            newpoint, newlabels = points[select], labels[select]
        if len(np.unique(newlabels)) < 2:
            return self
            

        self.fitted = NCA(random_state= self.seed)
        self.fitted.fit(self.preprocessor[np.array(newpoint)], newlabels) # perfect amai
        return self

    def transform(self, data):
        if self.fitted is None:
            return data
        return self.fitted.transform(data)
    
    def fit_transform(self, pairs, y, points, labels):
        if len(points) < 3:
            return np.copy(self.preprocessor)
        self.fit(pairs, y, points, labels)
        return self.transform(np.copy(self.preprocessor))
    
class KLMNN_wrapper(MetricLearner):
    def __init__(self, preprocessor=None, seed = 42):
        self.fitted = None
        self.seed = seed
        super().__init__(preprocessor) # TODO: dit uitbereiden

    def fit(self, pairs, y, points, labels):
        # als er labels minder dan 3 keer voorkomen, eruit halen ja
        unique, counts = np.unique(labels, return_counts=True)
        problem = unique[counts < 3]
        newpoint, newlabels = points, labels
        if (len(problem) > 0):
            select = np.invert(np.in1d(labels, problem))
            newpoint, newlabels = points[select], labels[select]
        if len(np.unique(newlabels)) < 2:
            return self
            

        self.fitted = KLMNN(kernel = "rbf")
        self.fitted.fit(self.preprocessor[np.array(newpoint)], newlabels) 
        return self

    def transform(self, data):
        if self.fitted is None:
            return data
        return self.fitted.transform(data)
    
    def fit_transform(self, pairs, y, points, labels):
        if len(points) < 3:
            return np.copy(self.preprocessor)
        self.fit(pairs, y, points, labels)
        return self.transform(np.copy(self.preprocessor))
    
class LMNN_wrapper(MetricLearner):
    def __init__(self, preprocessor=None, seed = 42):
        self.fitted = None
        self.seed = seed
        super().__init__(preprocessor) # TODO: dit uitbereiden

    def fit(self, pairs, y, points, labels):
        # als er labels minder dan 3 keer voorkomen, eruit halen ja
        unique, counts = np.unique(labels, return_counts=True)
        problem = unique[counts < 3]
        newpoint, newlabels = points, labels
        if (len(problem) > 0):
            select = np.invert(np.in1d(labels, problem))
            newpoint, newlabels = points[select], labels[select]
        if len(np.unique(newlabels)) < 2:
            return self
            

        self.fitted = LMNN(random_state= self.seed)
        self.fitted.fit(self.preprocessor[np.array(newpoint)], newlabels) # perfect amai
        return self

    def transform(self, data):
        if self.fitted is None:
            return data
        return self.fitted.transform(data)
    
    def fit_transform(self, pairs, y, points, labels):
        if len(points) < 3:
            return np.copy(self.preprocessor)
        self.fit(pairs, y, points, labels)
        return self.transform(np.copy(self.preprocessor))
    
    
class RCA_wrapper(MetricLearner):
    def __init__(self, preprocessor = None, n_components = None, kernel = False):
        self.ensemble = None
        n_comp = preprocessor.shape[1] # ff zo gedaan, maar best anders doen
        if n_comp > 5:
            n_comp = 5
        else:
            if not kernel:
                n_comp = 'mle'
        pre = KernelPCA(kernel='rbf',n_components = n_comp).fit_transform(preprocessor) if kernel else PCA(n_components = n_comp, svd_solver='full').fit_transform(preprocessor)
        super().__init__(pre)
    def fit(self, pairs, y, points, labels):
        blobs = createBlobs(pairs, y)
        constr = []
        indici = []
        i = 0
        for blob in blobs:
            indici.extend(blob)
            constr.extend([i]*len(blob))
            i+=1
        self.ensemble = RCA(preprocessor=self.preprocessor).fit(indici, constr) # ff wat hacky
        return self

    def transform(self, data):
        return self.ensemble.transform(np.copy(self.preprocessor)), self.affinity 

class Spectral(MetricLearner): # de effectieve transformatie doen, TODO: nog aanpassen
    def __init__(self, preprocessor=None, expand = False):
        super().__init__(preprocessor, expand)
    
    def fit(self, pairs, y, points, labels):
        if self.expand:
            pairs, y = expand(pairs, y)
        data = self.preprocessor
        n_neighbours = min(10, len(data))
        sp = SpectralClustering(n_clusters=1, eigen_solver="arpack", affinity='nearest_neighbors', n_neighbors=n_neighbours).fit(data)
        aff = csr_matrix((len(self.preprocessor), len(self.preprocessor)))
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
        return Z @ W
        # return W1 @ W @ W.T @ W1.T

def expand(pairs, y): # dit moeten kunnen toevoegen aan aan de index, of daar gewoon al direct expanden
    newpairs = []
    newy = []
    blobs = createBlobs(pairs[y == 1])
    for blob in blobs:
        new = list(itertools.combinations(blob, 2))
        newpairs.extend(new)
        newy.extend([1]*len(new))

    for cl in pairs[y == -1]:
        left = min(cl)
        right = max(cl)
        if ([left, right] in newpairs or [right, left] in newpairs):
            continue
        leftblob = [left]
        rightblob = [right]

        found = 0
        for blob in blobs:
            if left in blob:
                leftblob = blob
                found += 1
            if right in blob:
                rightblob = blob
                found += 1
            if found == 2:
                break
        
        newcl = np.transpose([np.tile(leftblob, len(rightblob)), np.repeat(rightblob, len(leftblob))])

        newpairs.extend(newcl.tolist())
        newy.extend([-1]*len(newcl))


    return np.sort(np.array(newpairs)), np.array(newy)


def createBlobs(must_links):
    blobs = []
    seen_indices = [] # deze zitten dus in ML blobs
    for ml in must_links:
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

        if ind1 in blob2:
            continue

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

    return blobs

