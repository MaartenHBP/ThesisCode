import abc

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn_extra.cluster import KMedoids

from noise_robust_cobras.metric_learning.module.metriclearners import *

# Deze class is voor als er ingewikkeldere dingen moeten gebeuren in hoe de metric wordt geleerd -> gaan aan de cluster class er ook toevoegen


class ClusterAlgorithm:
    @abc.abstractmethod
    def cluster(self, data, indices, k, ml, cl, seed=None, centers = None): # Added extra stuff for more advanced stuff
        pass

    def get_name(self):
        return type(self).__name__

    @classmethod
    def get_translated_constraints(cls, ml, cl, indices):
        """
            Note that this method is not automatically called!

            This method simply translates constraints between instance IDs in the full dataset
            to constraints between instance IDs in a subset of the dataset (denoted by indices)
            It only retains constraints for which both involved instances are in the subset of the dataset

        :param ml: a set of tuples of indices in the full dataset (each tuple denotes a must-link constraint)
        :param cl: a set of tuples of indices in the full dataset (each tuple denotes a cannot-link constraint)
        :param indices: a list of indices in the full dataset that denote the subset of the dataset to which the constraints need to be translated
        :return: the translated must-link and cannot-link constraints
        :rtype Tuple[Set[Tuple[int,int]],Set[Tuple[int,int]]]
        """
        filtered_ml = cls.__filter_constraint_set(ml, indices)
        filtered_cl = cls.__filter_constraint_set(cl, indices)
        translated_ml = cls.__translate_constraint_set(filtered_ml, indices)
        translated_cl = cls.__translate_constraint_set(filtered_cl, indices)
        return translated_ml, translated_cl

    @staticmethod
    def __filter_constraint_set(constraint_set, indices):
        filtered_set = set()
        for i1, i2 in constraint_set:
            if i1 in indices and i2 in indices:
                filtered_set.add((i1, i2))
        return filtered_set

    @staticmethod
    def __translate_constraint_set(constraint_set, indices):
        return set(
            (indices.index(ml1), indices.index(ml2)) for ml1, ml2 in constraint_set
        )


class KMeansClusterAlgorithm(ClusterAlgorithm):
    def __init__(self, n_runs=10, askExtraConstraints = False): # true for now
        self.n_runs = n_runs

    def cluster(self, data, indices, k, ml, cl, seed=None, centers = None, cobras = None):  

        init = 'k-means++' if centers is None else centers 
        
        if seed is not None:
            km = KMeans(k, n_init=self.n_runs, random_state=seed, init = init)
        else:
            km = KMeans(k, n_init=self.n_runs, init = init)

        # only cluster the given indices
        km.fit(data[indices, :])

        # return the labels as a list of integers
        return km.labels_.astype(np.int)#, None

# OBSOLETE
    
class KMeansITMLClusterAlgorithm(ClusterAlgorithm):  # TODO: aanpassen
    def __init__(self, percentage = 0.1, limit = 0.3, n_runs=10):
        self.percentage = percentage
        self.limit = limit
        self.n_runs = n_runs
        self.done = False

    def cluster(self, data, indices, k, ml, cl, seed=None, centers = None, cobras = None):
        pairs, labels = cobras.constraint_index.getLearningConstraints(local = indices, both = True)
        newdata = np.copy(data)
        if not self.done: 
            needed = math.floor(len(indices)*self.percentage*self.limit)
            extrapairs, extralabels = cobras.query_random_points(options = indices, count = needed)
            pairs = np.append(pairs, extrapairs, axis = 0)
            labels = np.append(labels, extralabels)
            newdata = ITML_wrapper(preprocessor=np.copy(data), seed=seed).fit_transform(pairs, labels, None, None)
            self.done = True

        

        # learn a bloody metric
        # if needed > 1:
        #     have = len(labels)
            
        #     if have < needed:
        #         extrapairs, extralabels = cobras.query_random_points(options = indices, count = needed - have)
        #         newpairs = np.append(pairs, extrapairs, axis = 0)
        #         newlabels = np.append(labels, extralabels)
        #     else:
        #         newpairs = np.copy(pairs)
        #         newlabels = np.copy(labels)

        


        init = 'k-means++' if centers is None else centers 
        
        if seed is not None:
            km = KMeans(k, n_init=self.n_runs, random_state=seed, init = init)
        else:
            km = KMeans(k, n_init=self.n_runs, init = init)

        # only cluster the given indices
        km.fit(newdata[indices, :])

        # return the labels as a list of integers
        return km.labels_.astype(np.int)#, None

    
class KMeansITMLClusterAlgorithmAbsolute(ClusterAlgorithm): 
    def __init__(self, start_amount = 30, n_runs=10):
        self.start_amount = start_amount
        self.n_runs = n_runs

    def cluster(self, data, indices, k, ml, cl, seed=None, centers = None, cobras = None):
        pairs, labels = cobras.constraint_index.getLearningConstraints(local = indices, both = True)
        needed = min(self.start_amount, math.ceil(len(indices)/2))# math.floor((len(indices)/len(data)) * self.start_amount)

        newdata = np.copy(data)

        # learn a bloody metric
        if needed > 1:
            have = len(labels)
            
            if have < needed:
                extrapairs, extralabels = cobras.query_random_points(options = indices, count = needed - have)
                newpairs = np.append(pairs, extrapairs, axis = 0)
                newlabels = np.append(labels, extralabels)
            else:
                newpairs = np.copy(pairs)
                newlabels = np.copy(labels)

            newdata = ITML_wrapper(preprocessor=data, seed=seed).fit_transform(newpairs, newlabels, None, None)


        init = 'k-means++' if centers is None else centers 
        
        if seed is not None:
            km = KMeans(k, n_init=self.n_runs, random_state=seed, init = init)
        else:
            km = KMeans(k, n_init=self.n_runs, init = init)

        # only cluster the given indices
        km.fit(newdata[indices, :])

        # return the labels as a list of integers
        return km.labels_.astype(np.int)#, None

# class SpectralClusterAlgorithm(ClusterAlgorithm):
#     def __init__(self, n_runs=10): #TODO: properties as class variables
#         self.n_runs = n_runs

#     def cluster(self, data, indices, k, ml, cl, seed=None, affinity = None, centers = None, distanceMatrix = None):

#         n_neighbours = min(10, len(indices)) # werkt niet voor split_level estimation

#         sp = SpectralClustering(n_clusters=k, eigen_solver="arpack", affinity='nearest_neighbors', random_state=seed, n_neighbors=n_neighbours).fit(data[indices, :]) if affinity is None else SpectralClustering(n_clusters=k, eigen_solver="arpack", affinity='precomputed', random_state=seed).fit(affinity[indices, :][:,indices])


#         # return the labels as a list of integers
#         return sp.labels_.astype(np.int)#, None


# class KMedoidsCLusteringAlgorithm(ClusterAlgorithm):
#     def __init__(self, n_runs=10):
#         self.n_runs = n_runs

#     def cluster(self, data, indices, k, ml, cl, seed=None, affinity = None, centers = None, distanceMatrix = None): #ml and cl not used
#         init = 'k-means++' if centers is None else centers 
        
#         if seed is not None:
#             km = KMedoids(n_clusters=k, random_state=seed, metric='precomputed') # precomputed distance
#         else:
#             km = KMedoids(n_clusters=k, metric='precomputed')

#         # only cluster the given indices
#         km.fit(distanceMatrix[indices, :][: ,indices])

#         # return the labels as a list of integers
#         return km.labels_.astype(np.int)#, indices[km.medoid_indices_]