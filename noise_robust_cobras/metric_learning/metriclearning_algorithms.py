# TODO, dit algoritme aanpassen

from abc import abstractmethod
import numpy as np
from metric_learn import *
from noise_robust_cobras.metric_learning.metricLearners import *

class MetricLearningAlgorithm: # abstract class
    def __init__(self, when: str = 'before_splitting') -> None:
        self.affinity = None # the result could be an affinity matrix, so set it
        self.when = when
    @abstractmethod
    def learn(self, cobras, current_superinstance, current_cluster):
        pass
        # returned momenteel nothing, zet gewoon wat variables aan

    def getConstraints(self, cobras, localData = None, both = False):
        '''
        localdata: which data te be taken in consideration
        both: both the points of constraints need te be in localdata
        '''
        pass

    def executeNow(self, when: str):
        '''
        when: when they want to execute the learn

        returns: if it is allowed to than call the learn function
        '''

        return self.when == when 

class BasicLearning(MetricLearningAlgorithm):
    def __init__(self, metric, when: str = 'before_splitting') -> None:
        super().__init__(when)
        self.metric = metric

    def learn(self, cobras, current_superinstance, current_cluster): # here comes the basic algorithm
        return super().learn(cobras, current_superinstance, current_cluster)

class LearnOnce(BasicLearning):
    def __init__(self, metric, when: str = 'before_splitting', once = False) -> None:
        super().__init__(metric, when)
        self.once = once
        self.finished = False

    def learn(self, cobras, current_superinstance, current_cluster): # here comes the one
        return super().learn(cobras, current_superinstance, current_cluster)

class IterationLearning(LearnOnce):
    def __init__(self, metric, when: str = 'before_splitting', once=False) -> None:
        super().__init__(metric, when, once)
        self.amount = 0
        self.count = 0
        
    def learn(self, cobras, current_superinstance, current_cluster):
        return super().learn(cobras, current_superinstance, current_cluster)

class QueriesLearning(LearnOnce):
    def __init__(self, metric, when: str = 'before_splitting', once=False, queriesNeeded = 0) -> None:
        super().__init__(metric, when, once)
        self.queriesNeeded = 0



# # Easy option is to transform the data and still work with euclidian distance (but does this suffice?)
# class MetricLearningAlgorithm:
#     @abstractmethod
#     def learn(self, cobras_cluster, local = None): # local zijn al de datapoints die deeluitmaken van de metric learning
#         pass
    
#     @abstractmethod
#     def transformData(self, data, local = None) :
#         pass

#     def addData(self, X):
#         self.X = np.copy(X)

#     def addTrainigIndices(self, indices):
#         self.trainigIndices = indices

# class SupervisedMetric(MetricLearningAlgorithm): 
#     def __init__(self, algo, steps = 0, queriesNeeded = 0, once = False):
#         self.algo = algo
#         self.steps = steps
#         self.count = 0
#         self.current = None
#         self.canTransform = False
#         self.queriesNeeded = queriesNeeded
#         self.once = once
#         self.done = False

#     def learn(self, cobras_cluster, local = None):
#         indices = np.array(self.trainigIndices)
#         if local:
#             indices = np.intersect1d(indices, np.array(local))

#         if self.done:
#             self.canTransform = False
#             return
#         if self.count < self.steps:
#             self.count+=1
#             self.canTransform = False
#             return
#         self.current = self.algo["value"](**self.algo["parameters"])
#         self.current.fit(np.copy(self.X[indices]), cobras_cluster.clustering.construct_cluster_labeling()[indices]) # ENKEL TRAININGSINDICES GEBRUIKEN

#         self.count = 0

#         self.canTransform = True

#         if self.once:
#             self.done = True


#     def transformData(self, data, local = None):
#         if self.current is None or not (self.canTransform): # if count is zero, you can safely transform data
#             return data
#         if local:
#             X = np.copy(data)
#             X[local] = self.current.transform(np.copy(data[local]))
#             return X
#         return self.current.transform(np.copy(data))

# class SemiSupervisedMetric(MetricLearningAlgorithm):
#     def __init__(self, algo, steps = 0, queriesNeeded = 0, once = False):
#         self.algo = algo
#         self.steps = steps
#         self.count = 0
#         self.current = None
#         self.canTransform = False
#         self.queriesNeeded = queriesNeeded
#         self.once = once
#         self.done = False

#     def learn(self, cobras_cluster, local = None):
#         if self.count < self.steps:
#             self.count+=1
#             return
#         if self.done:
#             self.canTransform = False
#             return
#         constraints = np.array(list(cobras_cluster.constraint_index.constraints))
#         if (constraints.shape[0] < 2):
#             return
#         self.current = self.algo["value"](preprocessor=np.copy(self.X),**self.algo["parameters"])
#         tuples = np.zeros((constraints.shape[0], 2), dtype = int)
#         y = np.zeros(constraints.shape[0], dtype = int)
#         indices = range(constraints.shape[0])
#         if (local):
#             localCheck = np.array(local)
#             indices = []
#         for i in range(constraints.shape[0]):
#             tup = constraints[i].to_tuple_b()
#             tuples[i] = tup[0:2]
#             y[i] = tup[2]
#             if local:
#                 a = np.isin(tuples[i],localCheck)
#                 if (len(a[a == True]) > 0):
#                     indices.append(i)
#         if len(indices) < 2:
#             self.canTransform = False
#             return

#         self.current.fit(tuples[indices], y[indices])
#         self.count=0

#         self.canTransform = True

#         if self.once:
#             self.done = True

#     def transformData(self, data, local = None):
#         if self.current is None or not (self.canTransform):
#             return data
#         if local:
#             X = np.copy(data)
#             X[local] = self.current.transform(np.copy(data[local]))
#             return X
#         return self.current.transform(np.copy(data))
    

# class EuclidianDistance(MetricLearningAlgorithm):
#     def __init__(self):
#         self.algo = None
#         self.current = None

#     def learn(self, cobras_cluster):
#         # no point in learning
#         return


#     def transformData(self, data, local = None):
#         return data

    # def getDistanceFunction():   # dit kan een mooiere oplossing zijn maar wordt wat messy

    #     def euclidianDistance(a, b):
    #         return np.linalg.norm(a - b)

    #     return euclidianDistance()

