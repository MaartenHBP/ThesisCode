from plots import IRAPlot
from pathlib import Path
import os
from sklearn.metrics import adjusted_rand_score
from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.noisy_labelquerier import ProbabilisticNoisyQuerier
from noise_robust_cobras.querier.labelquerier import LabelQuerier
from noise_robust_cobras.metric_learning.metriclearning_algorithms import SemiSupervisedMetric
import numpy as np

def varianceTest(path, day, data = None):

    # # variance test
    # varianceTest(path, currentDay)
    # import inquirer
    # questions = [
    # inquirer.List('size',
    #                 message="What size do you need?",
    #                 choices=['Jumbo', 'Large', 'Standard', 'Medium', 'Small', 'Micro'],
    #             ),
    # ]
    # answers = inquirer.prompt(questions)
    # print(answers["size"])
    # dit allemaal logi dat niet deel is van de variance test
    inp = None
    while True:
        path = Path('datasets/cobras-paper/').absolute()
        dir_list = os.listdir(path)
        # for i in range(89):
        #     str = str + "k"
        string = ""
        for i in dir_list:
            string = string + i[:len(string) - 5] + " | " 

        inp = input(string)
        if not inp:
            print("Variance analysis cancelled")
            return
        dataset_path = Path('datasets/cobras-paper/' + inp + ".data").absolute()

        if dataset_path.is_file():
            break

    i = 0
    while True:
        if i == 0:
            resulting_path = os.path.join(path, day + '_variancetest_' + inp)
        else: 
            resulting_path = os.path.join(path, day + '_variancetest_' + str(i) + '_' + inp)
        CHECK_FOLDER = os.path.isdir(resulting_path)
        if not CHECK_FOLDER:
            os.makedirs(resulting_path)
            break
            
        else:
            i += 1

    plt = IRAPlot("Variance test " + inp + " dataset") 

    # loading dataset
    dataset = np.loadtxt(dataset_path, delimiter=',')
    data = dataset[:, 1:]
    target = dataset[:, 0] 

    count = 500

    x = np.arange(count)
    x = x + 1

    for i in range(count):
        querier = LabelQuerier(None, target, i + 1)


        # make new COBRAS
        clusterer = COBRAS(correct_noise=False)
        all_clusters, runtimes, *_ = clusterer.fit(data, -1, None, querier)
        best_clustering = all_clusters[-1]
        runtime = runtimes[-1]

        plt.addPoint("without ml", i, adjusted_rand_score(target, best_clustering))
        
        querier = LabelQuerier(None, target, i + 1)

        # make new COBRAS
        clusterer = COBRAS(correct_noise=False, metric_algo=SemiSupervisedMetric())
        all_clusters, runtimes, *_ = clusterer.fit(data, -1, None, querier)
        best_clustering = all_clusters[-1]
        runtime = runtimes[-1]

        plt.addPoint("with ml", i, adjusted_rand_score(target, best_clustering))

        print(f"{(i + 1)/(count)*100:.1f} %", end="\r") # make a cool loading thingy
            

        



    plt.viewPlot()

        

# test = IRAPlot("test")

# test.addPoint("algo 1", 1, 3)
# test.addPoint("algo 1", 2, 10)
# test.addPoint("algo 1", 3, 13)
# test.addPoint("algo 1", 4, 36)
# test.addPoint("algo 1", 5, 39)
# test.addPoint("algo 1", 6, 60)
# test.addPoint("algo 1", 7, 40)

# test.addPoint("algo 2", 1, 33)
# test.addPoint("algo 2", 2, 40)
# test.addPoint("algo 2", 3, 50)
# test.addPoint("algo 2", 4, 51)
# test.addPoint("algo 2", 5, 51)
# test.addPoint("algo 2", 6, 8)
# test.addPoint("algo 2", 7, 13)

# test.viewPlot()
# test.viewPlot()

# test1 = IRAPlot("test")

# test1.addPoint("algo 1", 1, 3)
# test1.addPoint("algo 1", 2, 10)
# test1.addPoint("algo 1", 3, 13)
# test1.addPoint("algo 1", 4, 36)
# test1.addPoint("algo 1", 5, 39)
# test1.addPoint("algo 1", 6, 60)
# test1.addPoint("algo 1", 7, 40)

# test1.viewPlot()
# inp = input("$ ")
