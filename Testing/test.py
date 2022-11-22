# Importing pandas package
import json
from metric_learn import *
from metricalgos import *

dictionary = {
    "Cobras" : {
        "Description" : "Standard Cobras algorithm",
        "fitparam": {

        },
        "cobrasparam": {

        }
    },
    "CobrasPreproccesed" : {
        "Description": "Cobras but with data preprocessed",
        "fitparam": {
            "preprocessor": {
                "type": "supervised"
            },
            "preprocestraining": {
                "type": "Boolean"
            }
        },
        "cobrasparam": {

        }

    },
    "CobrasSupervised" : {
        "Description": "Cobras but with supervised clustering after each iteration",
        "fitparam": {
            "baseline": {
                "type": "Boolean"
            },
        },
        "cobrasparam": {
            "metric_algo": {
                "type": "class",
                "value": "SupervisedMetric",
                "parameters": {
                    "algo": {
                        "type": "supervised",
                    },
                    "steps": {
                        "type": "int",
                        "min": 0,
                        "max": 100
                    }
                }
            }
        }

    },
    "CobrasSemiSupervised" : {
        "Description": "Cobras but with supervised clustering after each iteration",
        "fitparam": {
            "baseline": {
                "type": "Boolean"
            },
        },
        "cobrasparam": {
            "metric_algo": {
                "type": "class",
                "value": "SupervisedMetric",
                "parameters": {
                    "algo": {
                        "type": "semisupervised",
                    },
                    "steps": {
                        "type": "int",
                        "min": 0,
                        "max": 100
                    }
                }
            }
        }

    }
}

def whatToAsk(algos):
    dictio = {}
    for key,values in algos.items():
        path, what = findAsk(values)
        dictio[key] = {"path": path, "what": what}
    return dictio

def findAsk(dictio):
    if "type" in dictio.keys():
        if "parameters" in dictio.keys():
            path, dicts = findAsk(dictio["parameters"])
            for i in path:
                i.insert(0, "parameters")
            return path, dicts
        else:
            return [[]], [dictio["type"]] 

    bigpath, bigdict = [], []
    for key,value in dictio.items():
        if type(value) is dict:
            path, dicts = findAsk(value)
            for i in path:
                i.insert(0, key)
            bigpath += path
            bigdict += dicts
    return bigpath, bigdict

# dictionary = {}

# supervised = {}
# semisupervised = {}

# for i in MetricAlgos.semisupervised:
#     semisupervised[i.__name__] = MetricAlgos.getArguments(i)

# for i in MetricAlgos.supervised:
#     supervised[i.__name__] = MetricAlgos.getArguments(i)

# dictionary["supervised"] = supervised
# dictionary["semisupervised"] = semisupervised

# dictionary = {
#     "test": ITML.__name__
# }
    
# with open("settings/algorithms.json", "w") as outfile:
#     json.dump(dictionary, outfile, indent=4)




# with open('settings/algorithms.json') as json_file:
#     data = json.load(json_file)

print(whatToAsk(dictionary))
#     print(type(data))
# print(eval(dictionary["test"]))
# def test_func(a = 4, b = 5):
#     print("The value of a is : " + str(a))
#     print("The value of b is : " + str(b))

# test_dict = {'a' : 1}
  
# # printing original dictionary
# print("The original dictionary is : " + str(test_dict))
  
# # Testing with default values 
# print("The default function call yields : ")
# test_func()

# test_func(b = 3,**test_dict)