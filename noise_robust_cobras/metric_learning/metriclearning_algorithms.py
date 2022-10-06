from abc import abstractmethod
import numpy as np
from metric_learn import ITML

# makkelijkste optie is al de data transformeren

class MetricLearningAlgorithm:
    @abstractmethod
    def learn(self, cobras_cluster):
        pass
    
    @abstractmethod
    def transformData(self) :
        pass

    def addData(self, X):
        self.X = np.copy(X)

class SemiSupervisedMetric(MetricLearningAlgorithm):
    def __init__(self):
        self.algo = ITML(preprocessor=np.copy(X))

    def learn(self, cobras_cluster):
        a = np.array(list(cobras_cluster.constraint_index.constraints))
        np.zeros((a.shape[0], 2))
        np.zeros(a.shape[0])
        self.algo = ITML()
        self.algo.fit()
    

class EuclidianDistance(MetricLearningAlgorithm):
    def __init__(self):
        self.algo = None

    def learn(self, cobras_cluster):
        constraints = np.array(list(cobras_cluster.constraint_index.constraints))
        print(constraints)
        tuples = np.zeros((constraints.shape[0], 2), dtype = int)
        y = np.zeros(constraints.shape[0], dtype = int)
        for i in range(constraints.shape[0]):
            tup = constraints[i].to_tuple_b()
            tuples[i] = tup[0:2]
            y[i] = tup[2]

        print(tuples)
        print(y)

        print(constraints)

    def transformData(self):
        return self.X

    # def getDistanceFunction():   # dit kan een mooiere oplossing zijn maar wordt wat messy

    #     def euclidianDistance(a, b):
    #         return np.linalg.norm(a - b)

    #     return euclidianDistance()

