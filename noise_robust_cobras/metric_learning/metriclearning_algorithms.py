from abc import abstractmethod
import numpy as np
from metric_learn import ITML
from metric_learn import MMC
from metric_learn import RCA
from metric_learn import SDML
from metric_learn import LMNN
from metric_learn import NCA

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

class SupervisedMetric(MetricLearningAlgorithm): # over alle mogelijke dingen loopen
    def __init__(self):
        self.algo = None
        self.count = 0

    def learn(self, cobras_cluster):
        # if self.count < 10:
        #     self.count+=1
        #     return
        self.algo = NCA(max_iter=100)
    

        self.algo.fit(np.copy(self.X), cobras_cluster.clustering.construct_cluster_labeling())

    def transformData(self):
        if self.algo is None:
            return self.X 
        return self.algo.transform(np.copy(self.X))

class SemiSupervisedMetric(MetricLearningAlgorithm):
    def __init__(self):
        self.algo = None
        self.count = 0

    def learn(self, cobras_cluster):
        # if self.count < 10:
        #     self.count+=1
        #     return
        constraints = np.array(list(cobras_cluster.constraint_index.constraints))
        if (constraints.shape[0] < 2):
            return
        self.algo = ITML(preprocessor=np.copy(self.X))
        tuples = np.zeros((constraints.shape[0], 2), dtype = int)
        y = np.zeros(constraints.shape[0], dtype = int)
        for i in range(constraints.shape[0]):
            tup = constraints[i].to_tuple_b()
            tuples[i] = tup[0:2]
            y[i] = tup[2]

        self.algo.fit(tuples, y)
        self.count=0

    def transformData(self):
        if self.algo is None:
            return self.X 
        return self.algo.transform(np.copy(self.X))
    

class EuclidianDistance(MetricLearningAlgorithm):
    def __init__(self):
        self.algo = None

    def learn(self, cobras_cluster):
        constraints = np.array(list(cobras_cluster.constraint_index.constraints))
        tuples = np.zeros((constraints.shape[0], 2), dtype = int)
        y = np.zeros(constraints.shape[0], dtype = int)
        for i in range(constraints.shape[0]):
            tup = constraints[i].to_tuple_b()
            tuples[i] = tup[0:2]
            y[i] = tup[2]


    def transformData(self):
        return self.X

    # def getDistanceFunction():   # dit kan een mooiere oplossing zijn maar wordt wat messy

    #     def euclidianDistance(a, b):
    #         return np.linalg.norm(a - b)

    #     return euclidianDistance()

