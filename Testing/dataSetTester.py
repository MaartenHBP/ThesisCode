# A file with functions to test on simple datasets
import numpy as np

def createDataset():
    """
    To create a dataset
    """
    print("dataset created")

def learnMetric(data, metricLearner):
    """
    metricLearner is of the form:
    {metric: <the class>, parameters: <the parameters>}

    retruns the metriclearner fitted on the data
    """
    print("metric learned")

def executeCobras(preprocessor = None, baseline = False, parameters = {}):
    "executes COBRAS and returns what COBRAS would return"
    print("Executing COBRAS")



