# Importing pandas package
import json
# from metric_learn import *
# from metricalgos import *

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

# print(whatToAsk(dictionary))
#     print(type(data))
# print(eval(dictionary["test"]))
# def test_func(a = 4, b = 5):
#     print("The value of a is : " + str(a))
#     print("The value of b is : " + str(b))

# test_dict = {'a' : 1, 'b': 8, 'c': 44}
  
# # printing original dictionary
# print("The original dictionary is : " + str(test_dict))
  
# # Testing with default values 
# print("The default function call yields : ")
# test_func()

# test_func(**test_dict)

# from metric_learn import RCA

# X = [[1.2, 7.5], [1.3, 1.5],
#          [6.4, 2.6], [6.2, 9.7],
#          [1.3, 4.5], [3.2, 4.6],
#          [6.2, 5.5], [5.4, 5.4]]
# chunks = [0, 0, 1, 1, 2, 2, 3, 3]

# rca = RCA()
# rca.fit(X, chunks)
from threading import Thread, Lock

from pylab import *
import math
class eventmanager:
    pressed = False
    colors = ['r', 'b']
    cin = 0
    pointsX = []
    pointsY = []
    classes = []
    pts = []
    drawncolors = []
    link = None
    erasing = False
    lock = Lock()

def eraser(event):
    eventmanager.lock.acquire()
    x,y = event.xdata,event.ydata
    tob = []
    print(len(eventmanager.pointsX))
    a = len(eventmanager.pointsX)
    for i in range(a):
        x1, y1 = eventmanager.pointsX[i], eventmanager.pointsY[i]
        if math.dist([x,y], [x1,y1]) < 0.01:
            tob.append(i)
    for d in tob:
        if a != len(eventmanager.pointsX):
            return
        eventmanager.pointsX.pop(d)
        eventmanager.pointsY.pop(d)
        eventmanager.classes.pop(d)
        eventmanager.drawncolors.pop(d)       
    clf()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    scatter(eventmanager.pointsX, eventmanager.pointsY, color = eventmanager.drawncolors)
    draw()
    eventmanager.lock.release()



def click(event):
    """If the left mouse button is pressed: draw a little square. """
    if event.button == 3:
        eraser(event)
        return
    eventmanager.pressed = True
    x,y = event.xdata,event.ydata
    clf()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    eventmanager.pointsX.append(x)
    eventmanager.pointsY.append(y)
    eventmanager.classes.append(eventmanager.cin)
    eventmanager.drawncolors.append(eventmanager.colors[eventmanager.cin])
    scatter(eventmanager.pointsX, eventmanager.pointsY, color = eventmanager.drawncolors)
    draw()

def loop(event):
    if event.button == 3:
        if (eventmanager.erasing):
            return
        eventmanager.erasing = True
        eraser(event)
        eventmanager.erasing = False
        return
    if not eventmanager.pressed:
        return
    """If the left mouse button is pressed: draw a little square. """
    x,y = event.xdata,event.ydata
    clf()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    eventmanager.pointsX.append(x)
    eventmanager.pointsY.append(y)
    eventmanager.classes.append(eventmanager.cin)
    eventmanager.drawncolors.append(eventmanager.colors[eventmanager.cin])
    scatter(eventmanager.pointsX, eventmanager.pointsY, color = eventmanager.drawncolors)
    draw()

def on_release(event):
    eventmanager.pressed = False

def left(event):
    eventmanager.cin = (eventmanager.cin + 1) % 2

def delete(event):
    eventmanager.pointsX.pop()
    eventmanager.pointsY.pop()
    eventmanager.drawncolors.pop()
    clf()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    scatter(eventmanager.pointsX, eventmanager.pointsY, color = eventmanager.drawncolors)
    draw()



plt.title('Create a new simple 2D dataset')

plt.xlim([0, 1])
plt.ylim([0, 1])

gca().set_autoscale_on(False)
connect('button_press_event',click)
connect('motion_notify_event',loop)
connect('button_release_event',on_release)
connect('scroll_event',left)
connect('key_press_event',delete)
annotate('Line Of Disadvantage', xy=(20, 1), xytext=(7, 3),
            arrowprops=dict(facecolor='black', shrink=0.05))
annotate('Most Disadvantaged',xy=(20, 1), xytext=(5, 5)),
annotate('Least Disadvantaged',xy=(20, 1), xytext=(70, 1)),

show()

inp = input('Save file(y)?: ')

if (inp == 'y'):
    inp2 = input('Name of the data: ')
    print(f"Saving the file {inp2}")
    from pathlib import Path
    data = np.column_stack((eventmanager.classes,eventmanager.pointsX,eventmanager.pointsY))
    dataset_path = Path(f'datasets/drawn/{inp2}.data').absolute()
    np.savetxt(dataset_path,data, delimiter=',')
    print(f"Data saved in file {inp2}.data")