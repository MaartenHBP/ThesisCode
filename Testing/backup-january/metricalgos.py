from metric_learn import *
import inspect
from abc import abstractmethod

class MetricAlgos:
    supervised = [NCA, LMNN, LFDA, MLKR, MMC_Supervised, ITML_Supervised, SDML_Supervised, RCA_Supervised]
    semisupervised = [ITML, SDML, RCA, MMC]
    @staticmethod
    def getAlg(kind):
        if not kind:
            return []
        if kind == 1:
            return MetricAlgos.semisupervised
        return MetricAlgos.supervised#[a.__name__ for a in MetricAlgos.supervised]
        
    @staticmethod
    def getArguments(a):
        dictio = {}
        b = inspect.signature(a)
        for k in b.parameters.values():
            dictio[k.name] = k.default

        return dictio

class MetricAlgo:
    @abstractmethod
    def __init__(self, preprocessor):
        self.X = preprocessor

    @abstractmethod
    def fit(tuples, y):
        pass

    def transform(data):
        return data

