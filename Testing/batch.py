from xmlrpc.client import Boolean, boolean
import pandas as pd

class Batch():
    def __init__(self, nameDataSet, nameAlgo, maxQ, runsPQ, crossFold: bool, metricPreprocessing: bool):
        self.nameDataSet = nameDataSet
        self.nameAlgo = nameAlgo
        self.crossFold = crossFold
        self.metricPreprocessing = metricPreprocessing
        self.maxQ = maxQ
        self.runsPQ = runsPQ
        self.results = pd.DataFrame()

    def chechIfRunned(self): # if true than you have the results already
        return False


    def saveResults():
        pass # save the results
        