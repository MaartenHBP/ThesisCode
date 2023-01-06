# A file with functions to test on simple datasets
import numpy as np

def createDataset():
    """
    To create a dataset
    """
    print("dataset created")

def learnMetric(constraints, metricLearner, data):
    """
    metricLearner is of the form:
    {metric: <the class>, parameters: <the parameters>}

    retruns the transformed data
    """
    print("metric learned")

def executeCobras(preprocessor = None, baseline = False, parameters = {}):
    "executes COBRAS and returns what COBRAS would return"
    print("Executing COBRAS")

def getConstraints(type, data):
    "Create constraints"
    if (type == "supervised"):
        print("supervised constraints")

    else:
        print("semisupervised cosntraints")



