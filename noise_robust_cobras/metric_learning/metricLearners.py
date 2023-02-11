from numpy.linalg import matrix_rank
from numpy import linalg as LA
from scipy.linalg import eigh
from sklearn.metrics import pairwise_distances
import math

from copy import deepcopy
from time import time

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors._ball_tree import BallTree
from sklearn.tree import DecisionTreeRegressor
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold._locally_linear import barycenter_kneighbors_graph

from noise_robust_cobras.metric_learning.knn import knn_error_score
from metric_learn import RCA

from sklearn.metrics import pairwise_kernels # can be very usefull
from scipy.spatial.distance import cdist

from scipy.optimize import minimize


from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

from scipy.sparse import csr_matrix

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

##########
# KITML 2#
##########

# class KITML2:
#     def __init__(self, preprocessor = None, alpha=1.0, gamma=1.0, max_iter=100):
#         self.alpha = alpha
#         self.gamma = gamma
#         self.max_iter = max_iter
#         self.preprocessor = preprocessor
        
#     def fit(self, pairs, y):

#         X = np.copy(self.preprocessor)
#         must_links = pairs[y == 1]
#         cannot_links = pairs[y == -1]
#         n_samples, n_features = X.shape
#         K = rbf_kernel(X, gamma=self.gamma)
#         K = csr_matrix(K)
        
#         if must_links is None:
#             must_links = []
#         if cannot_links is None:
#             cannot_links = []
        
#         def cost_function(A):
#             A = np.array(A).reshape((n_samples, n_samples))
#             A = csr_matrix(A)
#             KA = K.dot(A)
#             obj = np.sum(np.multiply(K, A)) - np.trace(KA)
#             obj = 0.5 * obj + 0.5 * self.alpha * np.sum(A ** 2)
            
#             for i, j in must_links:
#                 obj -= A[i, j]
                
#             for i, j in cannot_links:
#                 obj += A[i, j] ** 2
            
#             return obj
        
#         def grad_function(A):
#             A = np.array(A).reshape((n_samples, n_samples))
#             A = csr_matrix(A)
#             KA = K.dot(A)
#             grad = K - KA + self.alpha * A
            
#             for i, j in must_links:
#                 grad[i, j] -= 1
                
#             for i, j in cannot_links:
#                 grad[i, j] += 2 * A[i, j]
            
#             return grad.flatten()
        
#         A0 = np.zeros((n_samples, n_samples))
#         result = minimize(cost_function, A0.flatten(), method="trust-ncg", jac=grad_function,
#                           options={"maxiter": self.max_iter})
#         self.A_ = result.x.reshape((n_samples, n_samples))
#         return self
    
#     def transform(self, X):
#         K = rbf_kernel(X, gamma=self.gamma)
#         return K.dot(self.A_)

class KITML2:
    def __init__(self, preprocessor, n_components=2, reg=1e-5):
        self.X = preprocessor
        self.n_components = n_components
        self.reg = reg
        self.L = None
        
    def fit(self, pairs, y):

        must_links = pairs[y == 1]
        cannot_links = pairs[y == -1]

        n, d = self.X.shape
        n_ref, d_ref = self.X.shape
        
        K = rbf_kernel(self.X)
        H = np.eye(n_ref) - (1/n_ref) * np.ones((n_ref, n_ref))
        H = np.dot(H, H.T)
        P = np.dot(K, np.dot(H, K.T))
        
        if must_links is not None:
            for i, j in must_links:
                P[i, j] = 1e10
                P[j, i] = 1e10
        
        if cannot_links is not None:
            for i, j in cannot_links:
                P[i, j] = -1e10
                P[j, i] = -1e10
        
        def objective(L):
            L = L.reshape((n, self.n_components))
            obj = np.trace(np.dot(np.dot(L.T, P), L))
            obj += self.reg * np.linalg.norm(L, ord='fro') ** 2
            return obj
        
        L0 = np.random.randn(n * self.n_components)
        self.L = minimize(objective, L0).x.reshape((n, self.n_components))

        return self
        
    def transform(self, X=None):
        K = rbf_kernel(X)
        X_transformed = np.dot(K, self.L)
        return X_transformed

#########
# KITML #
#########

# def rbf_kernel(X, gamma=None):
#     """Compute the RBF kernel matrix of the input data."""
#     if gamma is None:
#         # default to 1 / n_features
#         gamma = 1.0 / X.shape[1]
#     pairwise_dists = cdist(X, X, "sqeuclidean")
#     K = np.exp(-gamma * pairwise_dists)
#     return K

class KITML:
    def __init__(self, preprocessor, default_cost=1):
        self.reg = 1e-5
        # self.kernel_func = kernel_func
        self.default_cost = default_cost
        self.alpha = None
        self.preprocessing = preprocessor
        self.n_components = 2
    
    def fit(self, pairs, y):
        must_link = pairs[y == 1]
        cannot_link = pairs[y == -1]
        X = np.copy(self.preprocessing)
        n_samples = X.shape[0]
        C = self.compute_cost_matrix(n_samples, must_link, cannot_link)
        K = polynomial_kernel(X)
        
        def obj_func(L0):
            alpha = L0.reshape((n_samples, self.n_components))
            return 0.5 * np.sum(np.dot(np.dot(alpha.T, K), alpha)) - np.sum(np.sum(alpha))
        
        cons = ({'type': 'eq', 'fun': lambda alpha: np.dot(C, alpha) - alpha},
                {'type': 'ineq', 'fun': lambda alpha: alpha})
        
        L0 = np.ones((n_samples, self.n_components)).ravel()
        res = minimize(obj_func, L0, options={"maxiter": 20}).x.reshape((n_samples, self.n_components))
        self.alpha = res
        return self
    
    def transform(self, X):
        K = polynomial_kernel(X)
        X_transformed = np.dot(K, self.alpha)
        print(X_transformed)
        return X_transformed
    
    def compute_cost_matrix(self, n_samples, must_link, cannot_link):
        C = np.ones((n_samples, n_samples)) * self.default_cost
        for i, j in must_link:
            C[i, j] = 0
            C[j, i] = 0
        for i, j in cannot_link:
            C[i, j] = 10000
            C[j, i] = 10000
        return C

#############
# Kernel ML #
#############
class kernel:
    def __init__(self, preprocessor = None, n_components = None, nb_neigh = 0):
        self.ensemble = None
        self.distance = pairwise_distances(preprocessor, metric='braycurtis')
        self.d = math.floor(preprocessor.shape[1]/2)
        # self.d = 2
        self.alph = 0.2 if self.d > 5 else 0.02
        self.k = 10
        self.prepocessor = preprocessor

    def fit(self, pairs ,y):
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
        return self.ensemble
    

#######
# RCA #
#######
class RCA_wrapper:
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
        return self.ensemble.transform(data)

#####################
# Spectralembedding #
#####################
class SpectralEmbedding_class:
    def __init__(self, preprocessor = None, nb_neigh = 10):
        self.ensemble = None
        self.preprocessor = preprocessor
        self.nb_neigh = nb_neigh
    def fit(self, pairs ,y):
        sp = SpectralClustering(n_clusters=2, eigen_solver="arpack", affinity='nearest_neighbors', n_neighbors= self.nb_neigh).fit(self.preprocessor)
        aff = sp.affinity_matrix_
        # test.coonstruct_nearest_graph()
        # T = test._W
        self.ensemble = SpectralEmbedding(eigen_solver="arpack", affinity='precomputed').fit_transform(aff)
        return self

    def transform(self, data):
        return self.ensemble


###############################
# Adapted spectral clustering #
###############################
# SpectralClustering(n_clusters=2,
        # eigen_solver="arpack", affinity='nearest_neighbors').fit(data)
class ssSpectralClust:
    def __init__(self, preprocessor = None, nb_neigh = 10):
        self.ensemble = None
        self.preprocessor = preprocessor
        self.nb_neigh = nb_neigh
    def fit(self, pairs ,y):
        sp = SpectralClustering(n_clusters=2, eigen_solver="arpack", affinity='nearest_neighbors', n_neighbors= self.nb_neigh).fit(self.preprocessor)
        aff = sp.affinity_matrix_
        yc = np.copy(y)
        yc[yc==-1] = 0
        aff[pairs[:,0], pairs[:,1]] = yc
        aff[pairs[:,1], pairs[:,0]] = yc
        self.ensemble = SpectralEmbedding(eigen_solver="arpack", affinity='precomputed').fit_transform(aff)
        return self

    def transform(self, data):
        return self.ensemble

#########
#gb_lmnn#
######### -> is supervised metric learner
"""
This is a supervised metric learning algorithm
Design from: https://github.com/iago-suarez/py-gb-lmnn
"""
class gb_lmnn_class:
    def __init__(self, k = 3, L = None, verbose=False, depth=4, n_trees=200, lr=1e-3, no_potential_impo=10, subsample_rate=1.0):
            self.k = k
            self.L = L
            self.verbose = verbose
            self.depth = depth
            self.n_trees = n_trees
            self.lr = lr
            self.no_potential_impo = no_potential_impo # dit is een belangrijke parameter
            self.subsample_rate = subsample_rate
            self.xval = None
            self.yval = None
            # voor transform
            self.ensemble = None
    def fit(self, X,y):
        self.ensemble = gb_lmnn(X, y, k = self.k, L = self.L, verbose=self.verbose, depth=self.depth, 
        n_trees=self.n_trees, lr=self.lr, no_potential_impo=self.no_potential_impo, subsample_rate=self.subsample_rate,
            xval=self.xval, yval=self.yval)
        return self

    def transform(self, data):
        return self.ensemble.transform(data)

########################
# Laplacian Eigenmaps  #
########################
class laplacian_semi:
    def __init__(self, dim:int = 2, k:int = 2, eps = None, graph:str = 'k-nearest', weights:str = 'heat kernel', 
                 sigma:float = 0.1, laplacian:str = 'unnormalized', opt_eps_jumps:float = 1.5, preprocessor = None):
            self.dim = dim
            self.k = k
            self.eps = eps
            self.graph = graph
            self.weights = weights
            self.sigma = sigma
            self.laplacian = laplacian
            self.opt_eps_jumps = opt_eps_jumps
            self.preprocessor = preprocessor

            self.ensemble = None
    def fit(self, pairs,y):
        self.ensemble = LE(self.preprocessor, self.dim, self.k, self.eps, self.graph, 
        self.weights, self.sigma, 
        self.laplacian, self.opt_eps_jumps, ml = pairs[y == 1], cl = pairs[y == -1]).transform()
        return self

    def transform(self, data):
        return self.ensemble




########################
#gb_lmnn implementation#
########################
"""
Pythonic implementation of the paper:
    Kedem, D., Tyree, S., Sha, F., Lanckriet, G. R., & Weinberger, K. Q. (2012).
    Non-linear metric learning. In NIPS (pp. 2573-2581).
"""
__author__ = "Iago Suarez"
__email__ = "iago.suarez.canosa@alumnos.upm.es"
class Ensemble:
    """Ensemble class that predicts based on the weighted sum of weak learners."""

    @property
    def n_wls(self):
        return len(self.weak_learners)

    def __init__(self, weak_learners=None, learning_rates=None, L=None):
        if weak_learners is None:
            weak_learners = []
        if learning_rates is None:
            learning_rates = []
        self.weak_learners = weak_learners
        self.learning_rates = learning_rates
        self.L = L

    def transform(self, X):
        assert len(self.weak_learners) > 0, "Error: The model hasn't been trained"

        # extract from ensemble
        label_length = self.weak_learners[0].n_outputs_

        # initialize predictions
        n = X.shape[0]
        if self.L is None:
            p = np.zeros((n, label_length))
        else:
            p = X @ self.L.T

        # compute predictions from trees
        for i in range(self.n_wls):
            p += self.learning_rates[i] * self.weak_learners[i].predict(X)
        return p


def find_target_neighbors(X, labels, K, n_classes):
    N, D = X.shape
    targets_ind = np.zeros((N, K), dtype=int)
    for i in range(n_classes):
        jj, = np.where(labels == i)
        # Samples of the class i
        Xu = X[jj]
        kdt = BallTree(Xu, leaf_size=50, metric='euclidean')
        targets = kdt.query(Xu, k=K + 1, return_distance=False)
        targets_ind[jj] = jj[targets[:, 1:]]

    return targets_ind


def find_random_target_neighbors(X, labels, K, n_classes):
    N, D = X.shape
    targets_ind = np.zeros((N, K), dtype=int)
    for i in range(n_classes):
        jj, = np.where(labels == i)
        random_targets = np.random.choice(jj, (len(jj), K))
        # Check if the random selection has some pair i-i
        colliding_elements = random_targets == jj[:, np.newaxis]
        n_colliding_elements = colliding_elements.sum()
        while n_colliding_elements > 0:
            # If so, replace these values
            random_targets[colliding_elements] = np.random.choice(jj, n_colliding_elements)
            colliding_elements = random_targets == jj[:, np.newaxis]
            n_colliding_elements = colliding_elements.sum()
        targets_ind[jj] = random_targets

    return targets_ind


def find_impostors(pred, labels, n_classes, no_potential_impo):
    N = len(pred)
    active = np.zeros((N, no_potential_impo), dtype=int)
    for i in range(n_classes):
        ii, = np.where(labels == i)
        pi = pred[ii]
        jj, = np.where(labels != i)
        pj = pred[jj]
        # Find the nearest neighbors using a BallTree
        kdt = BallTree(pj, leaf_size=50, metric='euclidean')
        hardest_examples = kdt.query(pi, k=no_potential_impo, return_distance=False)
        active[ii] = jj[hardest_examples]

    return active


def compute_loss(X, T, I, i, grad, margin=1.0):
    # compute distances to target neighbors
    targets_distance = cdist(X[i, np.newaxis], X[T], 'sqeuclidean').flatten()
    lossT = np.sum(targets_distance)
    # compute the influence of the target neighbors in the gradient
    grad[i] += np.sum(X[np.newaxis, i] - X[T], axis=0)
    grad[T] += X[T] - X[i]

    dists = cdist(X[i, np.newaxis], X[I], 'sqeuclidean').flatten()
    # Hinge loss
    lossI = np.maximum(0, 1 + targets_distance[:, np.newaxis] - dists).sum()
    #  compute distances to impostors
    for k, k_distance in enumerate(dists):  # For each impostor
        for j, j_distance in zip(T, targets_distance):  # For each target neighbor
            if j_distance > k_distance - margin:
                grad[i] += X[I[k]] - X[j]
                grad[j] += X[j] - X[i]
                grad[I[k]] += X[i] - X[I[k]]

    return 0.5 * (lossT + lossI)


def lmnn_obj_loss(pred, targets_ind, active_ind, margin=1.0):
    """
    Computes the hinge loss and its gradient for the formula (8) of Non-linear Metric Learning:
    .. math::
        \mathcal{L}(\phi)=\sum_{i, j: j \sim i}\left\|\phi\left(\mathbf{x}_{i}\right)-\phi\left(\mathbf{x}_{j}\right)
        \right\|_{2}^{2}+\mu \sum_{k: y_{i} \neq y_{k}}\left[1+\left\|\phi\left(\mathbf{x}_{i}\right)-\phi\left(
        \mathbf{x}_{j}\right)\right\|_{2}^{2}-\left\|\phi\left(\mathbf{x}_{i}\right)-\phi\left(\mathbf{x}_{k}\right)
        \right\|_{2}^{2}\right]_{+}
    :param pred: Array of floats with shape (n_final_dims, n_samples).
    The actual points X projected on the target low-dimensional space.
    :param targets_ind: Array of integers with shape (n_final_dims, n_samples).
    Indices of target neighbors, the ones that we want to keep close.
    :param active_ind: Array of integers with shape (N_IMPOSTORS, n_samples). Impostor indices.
    :return: The evaluation of the loss and its gradient
    """
    assert pred.ndim == 2 and targets_ind.ndim == 2 and active_ind.ndim == 2
    assert pred.shape[0] == targets_ind.shape[0] == active_ind.shape[0]

    n_samples, n_dims = pred.shape
    hinge, grad = np.zeros(n_samples), np.zeros(pred.shape)
    for i in range(n_samples):
        hinge[i] = compute_loss(pred, targets_ind[i], active_ind[i], i, grad, margin)

    return hinge, grad


def hinge_loss(X, target_ind, impostor_ind, mu=1.0, margin=1.0):
    n_samples, n_dims = X.shape
    hinge = np.zeros(n_samples)

    dists = squareform(pdist(X, 'sqeuclidean'))
    all_target_dist = np.take_along_axis(dists, target_ind, axis=1)
    all_impostors_dist = np.take_along_axis(dists, impostor_ind, axis=1)
    sum_target_dists = np.sum(all_target_dist, axis=1)

    for i in range(n_samples):
        lossI = np.maximum(0, margin + all_target_dist[i, :, np.newaxis] - all_impostors_dist[i]).sum()
        hinge[i] = sum_target_dists[i] + mu * lossI
    return hinge.mean()


def violating_hinge_loss(pred, target_ind, impostor_ind, margin=1.0):
    dists = squareform(pdist(pred, 'sqeuclidean'))
    all_target_dist = np.take_along_axis(dists, target_ind, axis=1)
    all_impostors_dist = np.take_along_axis(dists, impostor_ind, axis=1)

    violating = 0
    for i in range(len(pred)):
        violating += np.any(margin + all_target_dist[i, :, np.newaxis] - all_impostors_dist[i] > 0)

    return violating


def find_best_alpha(pred, wl_pred, target_ind, impostor_ind, loss_f=hinge_loss):
    local_diff_step = 1e-6
    alpha_interval = (-1, 2)
    alpha_interval_width = max(alpha_interval) - min(alpha_interval)

    while alpha_interval_width > 1e-8:
        alpha = (max(alpha_interval) + min(alpha_interval)) / 2.0
        aprox_gradient = (loss_f(pred + (alpha + local_diff_step) * wl_pred, target_ind, impostor_ind) -
                          loss_f(pred + (alpha - local_diff_step) * wl_pred, target_ind, impostor_ind)) / \
                         (2 * local_diff_step)
        if aprox_gradient.mean() > 0:
            # Move to the left
            alpha_interval = (alpha_interval[0], alpha)
        else:  # gradient < 0:
            # Move to the right
            alpha_interval = (alpha, alpha_interval[1])
        alpha_interval_width = max(alpha_interval) - min(alpha_interval)

    return alpha


def gb_lmnn(X, y, k, L, verbose=False, depth=4, n_trees=200, lr=1e-3, no_potential_impo=10, subsample_rate=1.0,
            xval=np.array([]), yval=np.array([])) -> Ensemble:
    """
    Nonlinear metric learning using gradient boosting regression trees.
    :param X: (NxD) is the input training data, 'labels' (1xn) the corresponding labels
    :param y: (N) is an initial linear transformation which can be learned using LMNN
    :param k: Number of nearest neighbours used to do the train step.
    :param L: (kxd) is an initial linear transformation which can be learned using LMNN.
    corresponds to a metric M=L'*L
    :param verbose: Displays the training evolution
    :param depth: Tree depth
    :param n_trees: number of boosted trees
    :param lr: learning rate for gradient boosting
    :param no_potential_impo: The number of potential impostors that will be used to pergorm the gradient computation.
    :param xval: The validation samples
    :param yval: The validation labels
    :return:
    """

    assert len(X) == len(y) and X.ndim == 2 and y.ndim == 1
    # assert len(xval) == 0 or (xval.ndim == 2 and yval.ndim == 1 and len(xval) == len(yval))
    # assert len(xval) == 0 or X.shape[1] == xval.shape[1]

    un, labels = np.unique(y), np.copy(y)
    if not np.alltrue(un == np.arange(len(un))):
        un2 = np.arange(len(un))
        for i in un2:
            labels[labels == un[i]] = i
        un = un2
    assert np.alltrue(un == np.arange(len(un))), "Error: labels should have format [1, 2, ..., C]"
    n_classes = len(un)

    use_validation = False
    pred = X #@ L.T
    pred_val = xval if use_validation else None
    if use_validation:
        tr_err, val_err = knn_error_score([], X, y, xval, yval, k=1)
        print("Initial Training error: {:.2f}%, Val. error: {:.2f}%".format(100 * tr_err, 100 * val_err))

    # Initialize some variables
    N, D = X.shape

    # find K target neighbors
    targets_ind = find_target_neighbors(X, labels, k, n_classes)

    # initialize ensemble (cell array of trees)
    ensemble = Ensemble(L=L)

    # initialize the lowest validation error
    lowest_val_err = np.inf
    best_ensemble = deepcopy(ensemble)
    margin = 1.0

    # Perform main learning iterations
    while ensemble.n_wls < n_trees:
        start = time()
        # Select potential imposters
        impostor_ind = find_impostors(pred, labels, n_classes, no_potential_impo)
        hinge, grad = lmnn_obj_loss(pred, targets_ind, impostor_ind, margin)
        cost = np.sum(hinge)

        # Determine if we are going to use subsample
        if subsample_rate == 1.0:
            subsample_ind = slice(None)
        else:
            subsample_ind = np.random.randint(0, N, int(N * subsample_rate))

        # Train the weak learner tree
        tree = DecisionTreeRegressor(max_depth=depth)
        tree.fit(X[subsample_ind], -grad[subsample_ind])
        wl_pred = tree.predict(X)

        alpha = lr  # * find_best_alpha(pred, wl_pred, targets_ind, impostor_ind)

        # Update predictions
        pred = pred + alpha * wl_pred

        # Add the tree and thew learning rate to the ensemble
        ensemble.weak_learners.append(tree)
        ensemble.learning_rates.append(alpha)

        # if iter % 10 == 0 and verbose:
        # Print out progress
        elapsed = time() - start
        iter = ensemble.n_wls + 1
        if verbose:
            print("Iteration {}: loss is {:.6f}, violating inputs: {}, alpha: {:.6f}, in {:.2f}s".format(
                iter, cost / N, violating_hinge_loss(pred, targets_ind, impostor_ind, margin), alpha, elapsed))

        # update best_ensemble of validation data
        if use_validation:
            pred_val = pred_val + alpha * tree.predict(xval)

            if iter % 5 == 0 or iter == (n_trees - 1):
                tr_err, val_err = knn_error_score([], pred, y, pred_val, yval, k=1)
                if verbose:
                    print("Iteration {}: Training error: {:.2f}%, Val. error: {:.2f}%".format(
                        iter, 100 * tr_err, 100 * val_err))

                if val_err <= lowest_val_err:
                    lowest_val_err = val_err
                    best_ensemble = deepcopy(ensemble)
                    if verbose:
                        print('--->\t\tBest validation error! :D')
    return ensemble


########################
#    Semi spectral     #
########################
import numpy as np
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import pairwise_distances
from matplotlib import pyplot as plt
import networkx as nx
import seaborn as sns
sns.set()
class semiSpectral():
    def __init__(preprocessor = None):
        pass

    def fit(X, constraints, ML, dimension = 2, weight = 0.5, cutoff = None): # potentieel hier al iets meer vragen
        W = pairwise_distances(X, metric="euclidean")
        max = W.max()
        vectorizer = np.vectorize(lambda x: weight*x/max)#1 if x < 5 else 0)
        W = np.vectorize(vectorizer)(W)

        for i in range(len(constraints)):
            if ML[i] == 1:
                W[constraints[i][0],constraints[i][1]] = 1
                W[constraints[i][1],constraints[i][0]] = 1
            else:
                W[constraints[i][0],constraints[i][1]] = 0
                W[constraints[i][1],constraints[i][0]] = 0

        # degree matrix
        D = np.diag(np.sum(np.array(W.todense()), axis=1))
        print('degree matrix:')
        print(D)
        # laplacian matrix
        L = D - W
        print('laplacian matrix:')
        print(L)

        e, v = np.linalg.eig(L)

        return e[:,2]



##### PLS WORK WITH MANIFOLDS WEET IK VEEL WAT IK AAN HET DOEN BEN #####
from sklearn.metrics import pairwise_distances
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import networkx as nx

class LE:
    
    def __init__(self, X:np.ndarray, dim:int, k:int = 2, eps = None, graph:str = 'k-nearest', weights:str = 'heat kernel', 
                 sigma:float = 0.1, laplacian:str = 'unnormalized', opt_eps_jumps:float = 1.5, ml = None, cl = None):
        """
        https://github.com/JAVI897/Laplacian-Eigenmaps
        LE object
        Parameters
        ----------
        
        X: nxd matrix
        
        dim: number of coordinates
        
        k: number of neighbours. Only used if graph = 'k-nearest'
        
        eps: epsilon hyperparameter. Only used if graph = 'eps'. 
        If is set to None, then epsilon is computed to be the 
        minimum one which guarantees G to be connected
        
        graph: if set to 'k-nearest', two points are neighbours 
        if one is the k nearest point of the other. 
        If set to 'eps', two points are neighbours if their 
        distance is less than epsilon
        
        weights: if set to 'heat kernel', the similarity between 
        two points is computed using the heat kernel approach.
        If set to 'simple', the weight between two points is 1
        if they are connected and 0 otherwise. If set to 'rbf'
        the similarity between two points is computed using the 
        gaussian kernel approach.
        
        sigma: coefficient for gaussian kernel or heat kernel
        
        laplacian: if set to 'unnormalized', eigenvectors are 
        obtained by solving the generalized eigenvalue problem 
        Ly = λDy where L is the unnormalized laplacian matrix.
        If set to 'random', eigenvectors are obtained by decomposing
        the Random Walk Normalized Laplacian matrix. If set to 
        'symmetrized', eigenvectors are obtained by decomposing
        the Symmetrized Normalized Laplacian
        
        opt_eps_jumps: increasing factor for epsilon
        """
        
        self.X = X
        self.dim = dim
        self.k = k
        self.eps = eps
        if graph not in ['k-nearest', 'eps']:
            raise ValueError("graph is expected to be a graph name; 'eps' or 'k-nearest', got {} instead".format(graph))
        self.graph = graph
        if weights not in ['simple', 'heat kernel', 'rbf']:
            raise ValueError("weights is expected to be a weight name; 'simple' or 'heat kernel', got {} instead".format(weights))
        self.weights = weights
        self.sigma = sigma
        self.n = self.X.shape[0]
        if laplacian not in ['unnormalized', 'random', 'symmetrized']:
            raise ValueError("laplacian is expected to be a laplacian name; 'unnormalized', 'random' or 'symmetrized', got {} instead".format(laplacian))
        self.laplacian = laplacian
        self.opt_eps_jumps = opt_eps_jumps
        if self.eps is None and self.graph == 'eps':
            self.__optimum_epsilon()
        self.ml = ml
        self.cl = cl
    
    def __optimum_epsilon(self):
        """
        Compute epsilon
        
        To chose the minimum epsilon which guarantees G to be 
        connected, first, epsilon is set to be equal to the distance 
        from observation i = 0 to its nearest neighbour. Then
        we check if the Graph is connected, if it's not, epsilon
        is increased and the process is repeated until the Graph
        is connected
        """
        dist_matrix = pairwise_distances(self.X)
        self.eps = min(dist_matrix[0,1:])
        con = False
        while not con:
            self.eps = self.opt_eps_jumps * self.eps
            self.__construct_nearest_graph()
            con = self.cc == 1
            print('[INFO] Epsilon: {}'.format(self.eps))
        self.eps = np.round(self.eps, 3)
    
    def __heat_kernel(self, dist):
        """
        k(x, y) = exp(- ||x-y|| / sigma )
        """
        return np.exp(- (dist*dist)/self.sigma)
    
    def __rbf(self, dist):
        """
        k(x, y) = exp(- (1/2*sigma^2) * ||x-y||^2)
        """
        return np.exp(- dist**2/ (2* (self.sigma**2) ) )
    
    def __simple(self, *args):
        return 1
    
    def __construct_nearest_graph(self):
        """
        Compute weighted graph G
        """
        similarities_dic = {'heat kernel': self.__heat_kernel,
                            'simple':self.__simple,
                            'rbf':self.__rbf}
        
        dist_matrix = pairwise_distances(self.X)
        if self.graph == 'k-nearest':
            nn_matrix = np.argsort(dist_matrix, axis = 1)[:, 1 : self.k + 1]
        elif self.graph == 'eps':
            nn_matrix = np.array([ [index for index, d in enumerate(dist_matrix[i,:]) if d < self.eps and index != i] for i in range(self.n) ])
        # Weight matrix
        self._W = []
        for i in range(self.n):
            w_aux = np.zeros((1, self.n))
            similarities = np.array([ similarities_dic[self.weights](dist_matrix[i,v]) for v in nn_matrix[i]] )
            np.put(w_aux, nn_matrix[i], similarities)
            self._W.append(w_aux[0])
        self._W = np.array(self._W)

        # print(self._W)

        if not self.ml is None:
            for i in range(len(self.ml)):
            #     if (self._W[self.ml[i][0],self.ml[i][1]] == 0):
            #         print("ja")
                self._W[self.ml[i][0],self.ml[i][1]] = 1
                self._W[self.ml[i][1],self.ml[i][0]] = 1
            for i in range(len(self.cl)):
                # if (self._W[self.ml[i][0],self.ml[i][1]] == 1):
                #     print("ja")
                self._W[self.cl[i][0],self.cl[i][1]] = 0
                self._W[self.cl[i][1],self.cl[i][0]] = 0
        # D matrix
        self._D = np.diag(self._W.sum(axis=1))
        # Check for connectivity
        self._G = self._W.copy() # Adjacency matrix
        self._G[self._G > 0] = 1
        G = nx.from_numpy_matrix(self._G)
        self.cc = nx.number_connected_components(G) # Multiplicity of lambda = 0
        if self.cc != 1:
            warnings.warn("Graph is not fully connected, Laplacian Eigenmaps may not work as expected")
            
    def __compute_unnormalized_laplacian(self):
        self.__construct_nearest_graph()
        self._L = self._D - self._W
        return self._L
    
    def __compute_normalized_random_laplacian(self):
        self.__construct_nearest_graph()
        self._Lr = np.eye(*self._W.shape) - (np.diag(1/self._D.diagonal())@self._W)
        return self._Lr
    
    def __compute_normalized_symmetrized_laplacian(self):
        self.__construct_nearest_graph()
        self.__compute_unnormalized_laplacian()
        d_tilde = np.diag(1/np.sqrt(self._D.diagonal()))
        self._Ls = d_tilde @ ( self._L @ d_tilde )
        return self._Ls
    
    def transform(self):
        """
        Compute embedding
        """
        
        m_options = {
            'unnormalized':self.__compute_unnormalized_laplacian,
            'random':self.__compute_normalized_random_laplacian,
            'symmetrized':self.__compute_normalized_symmetrized_laplacian
        }
        
        L = m_options[self.laplacian]()
        
        if self.laplacian == 'unnormalized':
            eigval, eigvec = eigh(L, self._D) # Generalized eigenvalue problem
        else:
            eigval, eigvec = np.linalg.eig(L)
            
        order = np.argsort(eigval)
        self.Y = eigvec[:, order[self.cc:self.cc+self.dim + 1]]
            
        return self.Y
    
    def plot_embedding_2d(self, colors, grid = True, dim_1 = 1, dim_2 = 2, cmap = None, size = (15, 10)):
        if self.dim < 2 and dim_2 <= self.dim and dim_1 <= self.dim:
            raise ValueError("There's not enough coordinates")
        
        # plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=size)
        plt.axhline(c = 'black', alpha = 0.2)
        plt.axvline(c = 'black', alpha = 0.2)
        if cmap is None:
            plt.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], c = colors)
            
        plt.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], c = colors, cmap=cmap)
        plt.grid(grid)
        if self.graph == 'k-nearest':
            title = 'LE with k = {} and weights = {}'.format(self.k, self.weights)
        else:
            title = 'LE with $\epsilon$ = {} and weights = {}'.format(self.eps, self.weights)
        plt.title(title)
        plt.xlabel('Coordinate {}'.format(dim_1))
        plt.ylabel('Coordinate {}'.format(dim_2))
        plt.show()
    
    def plot_embedding_3d(self, colors, grid = True, dim_1 = 1, dim_2 = 2, dim_3 = 3, cmap = None, size = (15, 10)):
        if self.dim < 3 and dim_2 <= self.dim and dim_1 <= self.dim and dim_3 <= self.dim:
            raise ValueError("There's not enough coordinates")
        
        # plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, projection="3d")
        if cmap is None:
            ax.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], self.Y[:, dim_3 - 1], c = colors)
        ax.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], self.Y[:, dim_3 - 1], c = colors, cmap = cmap)
        plt.grid(grid)
        ax.axis('on')
        if self.graph == 'k-nearest':
            title = 'LE with k = {} and weights = {}'.format(self.k, self.weights)
        else:
            title = 'LE with $\epsilon$ = {} and weights = {}'.format(self.eps, self.weights)
        plt.title(title)
        ax.set_xlabel('Coordinate {}'.format(dim_1))
        ax.set_ylabel('Coordinate {}'.format(dim_2))
        ax.set_zlabel('Coordinate {}'.format(dim_3))
        plt.show()

    def coonstruct_nearest_graph(self):
        """
        Compute weighted graph G
        """
        similarities_dic = {'heat kernel': self.__heat_kernel,
                            'simple':self.__simple,
                            'rbf':self.__rbf}
        
        dist_matrix = pairwise_distances(self.X)
        if self.graph == 'k-nearest':
            nn_matrix = np.argsort(dist_matrix, axis = 1)[:, 1 : self.k + 1]
        elif self.graph == 'eps':
            nn_matrix = np.array([ [index for index, d in enumerate(dist_matrix[i,:]) if d < self.eps and index != i] for i in range(self.n) ])
        # Weight matrix
        self._W = []
        for i in range(self.n):
            w_aux = np.zeros((1, self.n))
            similarities = np.array([ similarities_dic[self.weights](dist_matrix[i,v]) for v in nn_matrix[i]] )
            np.put(w_aux, nn_matrix[i], similarities)
            self._W.append(w_aux[0])
        self._W = np.array(self._W)

        # print(self._W)

        if not self.ml is None:
            for i in range(len(self.ml)):
            #     if (self._W[self.ml[i][0],self.ml[i][1]] == 0):
            #         print("ja")
                self._W[self.ml[i][0],self.ml[i][1]] = 1
                self._W[self.ml[i][1],self.ml[i][0]] = 1
            for i in range(len(self.cl)):
                # if (self._W[self.ml[i][0],self.ml[i][1]] == 1):
                #     print("ja")
                self._W[self.cl[i][0],self.cl[i][1]] = 0
                self._W[self.cl[i][1],self.cl[i][0]] = 0
        # D matrix
        self._D = np.diag(self._W.sum(axis=1))
        # Check for connectivity
        self._G = self._W.copy() # Adjacency matrix
        self._G[self._G > 0] = 1
        G = nx.from_numpy_matrix(self._G)
        self.cc = nx.number_connected_components(G) # Multiplicity of lambda = 0
        if self.cc != 1:
            warnings.warn("Graph is not fully connected, Laplacian Eigenmaps may not work as expected")

