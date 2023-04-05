from noise_robust_cobras.querier.querier import Querier, MaximumQueriesExceeded
import numpy as np
import random
from itertools import combinations


class LabelQuerier(Querier):
    """
        A querier which answers each query correctly
    """

    def __init__(self, logger, labels, maximum_number_of_queries):
        super(LabelQuerier, self).__init__(logger)
        self.labels = labels
        self.max_queries = maximum_number_of_queries
        self.queries_asked = 0

    def _query_points(self, idx1, idx2):
        if self.max_queries is not None and self.queries_asked >= self.max_queries:
            raise MaximumQueriesExceeded
        self.queries_asked += 1
        return self.labels[idx1] == self.labels[idx2]

    def query_limit_reached(self):
        if self.max_queries is None:
            return False
        return self.queries_asked >= self.max_queries
    
    ###################
    # Metric Learning #
    ###################

    def getConstraints(self, points):
        return np.array(self.labels)[np.array(points)]

    def getRandomConstraints(self, nbConstraints):
        if nbConstraints == 0:
            return [], []
        indices = np.arange(len(self.labels))
        all_pairs = np.array(list(combinations(indices, 2)))

        indi = np.arange(len(all_pairs))
        random.shuffle(indi)

        pairs =  all_pairs[indi][:nbConstraints]

        constraints = np.ones(nbConstraints)

        constraints[self.labels[pairs[:,0]] != self.labels[pairs[:, 1]]] = -1

        return pairs, constraints
    
    def getRandomLabels(self, nbContraints):
        if nbContraints == 0:
            return [], []
        indices = np.arange(len(self.labels))

        random.shuffle(indices)

        indi = indices[:nbContraints]

        return indi, np.array(self.labels)[indi]  

    
    def checkConstraints(self,constraints, y):
        for i in range(len(constraints)):
            if y[i] == 1 and self.labels[constraints[i][0]] != self.labels[constraints[i][1]]:
                return False
            if y[i] == -1 and self.labels[constraints[i][0]] == self.labels[constraints[i][1]]:
                return False
        return True
    
    def createCopy(self, maxQueries):
        return LabelQuerier(None, self.labels, maxQueries)
