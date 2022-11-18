from metric_learn import *
import inspect

class MetricAlgos:

    supervised = [NCA]
    semisupervised = [ITML]
    @staticmethod
    def getAlg(kind):
        if not kind:
            return []
        if kind == 1:
            return MetricAlgos.semisupervised
        return MetricAlgos.supervised#[a.__name__ for a in MetricAlgos.supervised]
        
    @staticmethod
    def getArguments(a):
        keys, _, _, values = inspect.getargvalues(a)
        kwargs = {}
        for key in keys:
            if key != 'self':
                kwargs[key] = values[key]
        return kwargs