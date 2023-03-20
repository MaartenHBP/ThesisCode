# TODO, dit algoritme aanpassen

from abc import abstractmethod
import numpy as np
from noise_robust_cobras.metric_learning.metriclearning import *


class MetricLearningAlgorithm: # abstract class
    def __init__(self, when: str = 'begin') -> None:
        self.transformed = None
        self.orginal = None
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


