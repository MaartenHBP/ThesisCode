from xmlrpc.client import Boolean, boolean
import pandas as pd
import os
from pathlib import Path

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
        path = Path('batches/' + self.getName()).absolute()
        if os.path.exists(path):
            newPath = Path('batches/' + self.newName()).absolute()
            os.rename(path, newPath)
            # self.results = pd.read_csv(path) #TODO: als er een test is gedaan met meer maxQ dan deze ndig heeft kunnen de resultaten ook gebruikt worden
            return True
        return False

    def saveResults(self): # save die shit direct als er gerunt is!
        self.results.to_csv('batches/' + self.getName())

    def getName(self): # algo_dataset_preproccesed_maxQ_runsPQ_crossfold
        string = self.nameAlgo + "_" + self.nameDataSet
        if self.metricPreprocessing:
            string = string + "_"  + "preprocessed"
        string = string + "_"  + str(self.maxQ) + "_" + str(self.runsPQ)
        if self.crossFold:
            string = string + "_" + "crossfold"
        return string

    def newName(self):
        string = self.nameAlgo 
        if self.metricPreprocessing:
            string = string +  "_" + "preprocessed"
        string = string + "_"  + self.nameDataSet
        string = string + "_"  + str(self.maxQ) + "_" + str(self.runsPQ)
        if self.crossFold:
            string = string + "_" + "crossfold"
        return string