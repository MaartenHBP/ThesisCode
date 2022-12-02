from abc import abstractmethod
import numpy as np
from metric_learn import *
# Easy option is to transform the data and still work with euclidian distance (but does this suffice?)

class MetricLearningAlgorithm:
    @abstractmethod
    def learn(self, cobras_cluster): # dit nog aanpassen
        pass
    
    @abstractmethod
    def transformData(self) :
        pass

    def addData(self, X):
        self.X = np.copy(X)

class SupervisedMetric(MetricLearningAlgorithm):
    def __init__(self, algo, steps = 0):
        self.algo = algo
        self.steps = steps
        self.count = 0
        self.current = None

    def learn(self, cobras_cluster):
        if self.count < self.steps:
            self.count+=1
            return
        self.current = self.algo["value"](**self.algo["parameters"])
        self.current.fit(np.copy(self.X), cobras_cluster.clustering.construct_cluster_labeling())

        self.count = 0

    def transformData(self):
        if self.current is None:
            return self.X 
        return self.current.transform(np.copy(self.X))

class SemiSupervisedMetric(MetricLearningAlgorithm):
    def __init__(self, algo, steps = 0):
        self.algo = algo
        self.steps = steps
        self.count = 0
        self.current = None

    def learn(self, cobras_cluster):
        if self.count < self.steps:
            self.count+=1
            return
        constraints = np.array(list(cobras_cluster.constraint_index.constraints))
        if (constraints.shape[0] < 2):
            return
        self.current = self.algo["value"](preprocessor=np.copy(self.X),**self.algo["parameters"])
        tuples = np.zeros((constraints.shape[0], 2), dtype = int)
        y = np.zeros(constraints.shape[0], dtype = int)
        for i in range(constraints.shape[0]):
            tup = constraints[i].to_tuple_b()
            tuples[i] = tup[0:2]
            y[i] = tup[2]

        self.current.fit(tuples, y)
        self.count=0

    def transformData(self):
        if self.current is None:
            return self.X 
        return self.current.transform(np.copy(self.X))
    

class EuclidianDistance(MetricLearningAlgorithm):
    def __init__(self):
        self.algo = None
        self.current = None

    def learn(self, cobras_cluster):
        # no point in learning
        return


    def transformData(self):
        return self.X

    # def getDistanceFunction():   # dit kan een mooiere oplossing zijn maar wordt wat messy

    #     def euclidianDistance(a, b):
    #         return np.linalg.norm(a - b)

    #     return euclidianDistance()

