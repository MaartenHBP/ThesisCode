# TODO, dit algoritme aanpassen

from abc import abstractmethod
import numpy as np
from noise_robust_cobras.metric_learning.metriclearning import *


class MetricLearningAlgorithm: # abstract class
    def __init__(self, when: str = 'begin') -> None:
        self.transformed = None # -> this way just gives a little more power for controlling
        self.affinity = None # for when we need the affinity matrix (spectral transformation could give both, it then depends which type of algorithm uses the metric learn results)
        ######################## -> affinity is always for the whole dataset (ga ervan uit dat dit zo is bij de metriclearners)
        self.orginal = None
        ########################
        self.when = when
    @abstractmethod
    def learn(self, cobras, current_superinstance, current_cluster):
        pass
        # returned momenteel nothing, zet gewoon wat variables aan

    # def getConstraints(self, cobras, localData = None, both = False): -> zie de constraint index
    #     '''
    #     localdata: which data te be taken in consideration
    #     both: both the points of constraints need te be in localdata
    #     '''
    #     pass

    def executeNow(self, when: str):
        '''
        when: when they want to execute the learn

        returns: if it is allowed to than call the learn function
        '''

        return self.when == when 

    def setOriginal(self, X):
        self.orginal = np.copy(X)

    def getDataForClustering(self):
        trans = np.copy(self.transformed) if self.transformed is not None else None
        return trans, self.affinity

    def isDone(self):
        return False

##################
# Simple testing #
##################

class SimpleLearning(MetricLearningAlgorithm):
    def __init__(self, metric = None, when: str = 'begin', nbConstraints = 100) -> None:
        super().__init__(when)
        self.metric = metric
        self.nbConstraints = nbConstraints
        self.done = False


    def getDataForClustering(self):
        return None, None # is het veiligste

    def learn(self, cobras, current_superinstance, current_cluster): # here comes the basic algorithm
        if not self.metric or self.done:
            return False

        pairs, constraints = None, None
        if (self.metric["value"].__name__ == "NCA_wrapper"): # nbconstraints is percemtage to keep for NCA
            select = np.random.choice(len(self.orginal), math.floor((self.nbConstraints/100)*len(self.orginal)), replace=False)
            pairs = np.copy(self.orginal[select])
            constraints = cobras.querier.labels[select]
        else:
            pairs, constraints = cobras.querier.getRandomConstraints(self.nbConstraints)  if self.when == 'initial' else cobras.constraint_index.getLearningConstraints()

        if len(pairs) >= self.nbConstraints or self.when == 'end' or self.when == 'initial':
            # pairs, constraints = expand(pairs, constraints)

            # if not cobras.querier.checkConstraints(pairs, constraints):
            #     print("OEI OEI OEI")
            # else:
            #     print("tis inrode zenne")
            self.done = True
            self.learner = self.metric["value"](preprocessor = np.copy(self.orginal),**self.metric["parameters"])
            self.transformed, self.affinity = self.learner.fit(pairs, constraints).transform(np.copy(self.orginal))
            cobras.data = np.copy(self.transformed)
        return False

    def isDone(self):
        return self.done

#################
# basic options #
#################

class BasicLearning(MetricLearningAlgorithm):
    def __init__(self, metric = None, local = False, # rebuilder has to be created already
    cluster = False, both = False, useTransformed =  True, iterative = False, when: str = 'begin') -> None:
        super().__init__(when)
        self.metric = metric
        ###########################
        self.local = local # this is for the splitting phase
        self.cluster = cluster
        self.both = both
        ###########################
        self.useTransformed = useTransformed
        self.iterative = iterative
        ###########################

    def getDataForClustering(self):
        if self.local or self.cluster:
            return super().getDataForClustering()
        return None, None # cannot use the results from here

    def useResultsForSplitting(self): # wordt niet gebruikt momenteel
        return self.local or self.cluster

    def learn(self, cobras, current_superinstance, current_cluster): # here comes the basic algorithm
        # first get the constraints
        local = current_cluster.get_all_points() if self.cluster else (current_superinstance.indices if self.local else None)
        if self.cluster: # superinstance was er al uitgehaald
                local.extend(current_superinstance.indices)
        if (self.metric["value"].__name__ == "NCA_wrapper"):
            pairs = np.copy(self.orginal)
            constraints = cobras.querier.labels 
            if local is not None:
                constraints = constraints[local]
                pairs = pairs[local]
        else: 
            pairs, constraints = cobras.constraint_index.getLearningConstraints(local, self.both)

        # print(f"amount of constraints = {len(pairs)}")
        # if len = 0 return and do nothing
        result = False
        if self.metric:
            self.learner = self.metric["value"](preprocessor = np.copy(self.orginal),**self.metric["parameters"]) # we hebben wrappers woehoe
            self.transformed, self.affinity = self.learner.fit(pairs, constraints,local).transform(np.copy(self.orginal))
            if self.useTransformed and not self.local and not self.cluster:
                cobras.data = np.copy(self.transformed)
            if self.iterative:
                self.orginal = np.copy(self.transformed)
        if cobras.rebuilder: # COBRAS has a rebuilder
            data = self.orginal if self.transformed is None else self.transformed # normally work with transformed
            result = cobras.rebuilder.rebuildInstances(cobras, data, self.affinity)
        return result

class LearnOnce(BasicLearning):
    def __init__(self, metric = None, local = False, 
    cluster = False, both = False, useTransformed =  True,  iterative = False, when: str = 'begin', once = False) -> None:
        super().__init__(metric, local, cluster, both, useTransformed, iterative, when)
        self.once = once
        self.finished = False

    def learn(self, cobras, current_superinstance, current_cluster): # here comes the one
        if not self.finished:
            self.finished = self.once
            return super().learn(cobras, current_superinstance, current_cluster)
        return False
            

class IterationLearning(LearnOnce):
    def __init__(self, metric = None, local = False, 
    cluster = False, both = False, useTransformed =  True, iterative = False, when: str = 'begin', once=False, amount = 0) -> None:
        super().__init__(metric, local, cluster, both, useTransformed, iterative, when, once)
        self.amount = amount
        self.count = 0
        
    def learn(self, cobras, current_superinstance, current_cluster):
        if self.count < self.amount:
            self.count += 1
            return False
        else:
            self.count = 0
            return super().learn(cobras, current_superinstance, current_cluster)

class QueriesLearning(LearnOnce): # deze is nog niet af
    def __init__(self, metric = None, local = False, 
    cluster = False,both = False, useTransformed =  True, iterative = False, when: str = 'begin', once=False, queriesNeeded = 20, restartCouning = True) -> None:
        super().__init__(metric, local, cluster, both, useTransformed, iterative, when, once)
        self.queriesNeeded = queriesNeeded
        self.seen = 0 
        self.restartCouning = restartCouning

    def learn(self, cobras, current_superinstance, current_cluster): # dubbele berekening van constraints en als het aantal in de cluster afhangt wordt hier niet naar gekeken
        if len(cobras.constraint_index.constraints) - self.seen < self.queriesNeeded:
            return False
        if self.restartCouning:
            self.seen += self.queriesNeeded
        return super().learn(cobras, current_superinstance, current_cluster)

#######################
# Euclidian distance  #
#######################
class EuclidianDistance(MetricLearningAlgorithm):
    def learn(self, cobras, current_superinstance, current_cluster):
        return False
    def setOriginal(self, X):
        super().setOriginal(X)
        self.transformed = np.copy(self.orginal) # als het later voor clusteren wordt gebruikt

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

