from metric_learn import *
import inspect

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
