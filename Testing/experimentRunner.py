from math import sqrt
from plots import IRAPlot
from pathlib import Path
import os
from sklearn.metrics import adjusted_rand_score
from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.noisy_labelquerier import ProbabilisticNoisyQuerier
from noise_robust_cobras.querier.labelquerier import LabelQuerier
from noise_robust_cobras.metric_learning.metriclearning_algorithms import SemiSupervisedMetric
import numpy as np 
import scipy

class ExperimentRunner:
    def __init__(self, name: str, path, day) -> None:
        self.name = name
        self.datasets = dict()
        self.plotmakers = dict()

        # i = 0
        # while True:
        #     if i == 0:
        #         resulting_path = os.path.join(path, day + "_" + self.name)
        #     else: 
        #         resulting_path = os.path.join(path, day + "_" + self.name + '_' + str(i) )
        #     CHECK_FOLDER = os.path.isdir(resulting_path)
        #     if not CHECK_FOLDER:
        #         os.makedirs(resulting_path)
        #         break
                
        #     else:
        #         i += 1
        #self.path = resulting_path
        self.day = day

    def loadDataSets(self, datasets = None):
        signal = False
        if datasets is None:
            inp = None
            while True:
                path = Path('datasets/cobras-paper/').absolute()
                dir_list = os.listdir(path)
                string = ""
                for i in dir_list:
                    string = string + i[:len(string) - 5] + " | " 

                inp = input(string)
                if not inp:
                    return signal
                dataset_path = Path('datasets/cobras-paper/' + inp + ".data").absolute() # brow er wordt nergens gecheched of die al bestaat
                samples = self.datasets.get(inp, None)
                if samples is not None:
                    print("bestaat al kut")
                    continue
                signal = True
                dataset = np.loadtxt(dataset_path, delimiter=',')
                data = dataset[:, 1:]
                target = dataset[:, 0]
                self.datasets[inp] = {
                                        "data": data,
                                        "target": target,
                                        }

    def run(self, ARI = True, time = False, plot = False, 
            maxQ = 100, runsPQ = 1, CI = False, metric = True, #nu nog true maar moet een lijst van algo's worden
            normal = True): # returns de COBRAS indien een run
            self.plotmakers["ARI"] = IRAPlot("ARI plot")
            self.plotmakers["time"] = IRAPlot("time plot")
            if plot:
                self.plotmakers["plot"] = IRAPlot("plot")



            for name,samples in self.datasets.items():
                average = {"S1": np.zeros(maxQ), "S2": np.zeros(maxQ), "mu": np.zeros(maxQ)}
                for j in range(runsPQ):
                    querier = LabelQuerier(None, samples["target"], maxQ)
                    clusterer = COBRAS(correct_noise=False)
                    all_clusters, runtimes, *_ = clusterer.fit(samples["data"], -1, None, querier)
                    # best_clustering = all_clusters[-1]
                    # runtime = runtimes[-1]
                    IRA = np.array([adjusted_rand_score(samples["target"], clustering) for clustering in all_clusters])
                    
                    average["S1"] += IRA
                    average["S2"] += IRA**2
    
                    print(f"{(j + 1)/(runsPQ)*100:.1f} %", end="\r")

                average["mu"] = average["S1"]/runsPQ
                seNormal = np.sqrt((runsPQ*average["S2"] - average["S1"]**2)/(runsPQ*(runsPQ-1)))
                tp = scipy.stats.t.ppf((1 + 0.95) / 2., runsPQ - 1)
                hNormal = seNormal * tp

                self.plotmakers["ARI"].addPoint(name + " no metric learning", average["mu"])
                self.plotmakers["ARI"].addPoint(name + " upper", average["mu"] + hNormal)
                self.plotmakers["ARI"].addPoint(name + " lower", average["mu"] - hNormal)

            # for name,samples in self.datasets.items():
            #     for i in range(maxQ):
            #         averageNormal = {"S1": 0, "S2": 0, "time": 0, "mu": 0}
            #         averageMetric = {"S1": 0, "S2": 0, "time": 0, "mu": 0}
            #         for j in range(runsPQ):
            #             if normal:
            #                 querier = LabelQuerier(None, samples["target"], i + 1)
            #                 clusterer = COBRAS(correct_noise=False)
            #                 all_clusters, runtimes, *_ = clusterer.fit(samples["data"], -1, None, querier)
            #                 best_clustering = all_clusters[-1]
            #                 runtime = runtimes[-1]
            #                 averageNormal["S1"] += adjusted_rand_score(samples["target"], best_clustering)
            #                 averageNormal["S2"] += adjusted_rand_score(samples["target"], best_clustering)**2
            #                 averageNormal["time"] += (runtime - averageNormal["time"])/(j + 1)
            #                 print(f"{((i*runsPQ + j) * maal + 1)/(totalRuns)*100:.1f} %", end="\r")
            #             if metric:
            #                 querier = LabelQuerier(None, samples["target"], i + 1)
            #                 clusterer = COBRAS(correct_noise=False, metric_algo=SemiSupervisedMetric())
            #                 all_clusters, runtimes, *_ = clusterer.fit(samples["data"], -1, None, querier)
            #                 best_clustering = all_clusters[-1]
            #                 runtime = runtimes[-1]
            #                 averageMetric["S1"] += adjusted_rand_score(samples["target"], best_clustering)
            #                 averageMetric["S2"] += adjusted_rand_score(samples["target"], best_clustering)**2
            #                 averageMetric["time"] += (runtime - averageMetric["time"])/(j + 1)
            #                 print(f"{((i*runsPQ + j) * maal + 2)/(totalRuns)*100:.1f} %", end="\r")
            #         averageNormal["mu"] = averageNormal["S1"]/runsPQ
            #         averageMetric["mu"] = averageMetric["S1"]/runsPQ
            #         seNormal = sqrt((runsPQ*averageNormal["S2"] - averageNormal["S1"]**2)/(runsPQ*(runsPQ-1)))
            #         seMetric = 0 # nog ff niet nodig
            #         if normal:
            #             self.plotmakers["ARI"].addPoint(name + " no metric learning", i, averageNormal["mu"])
            #             self.plotmakers["time"].addPoint(name + " no metric learning", i, averageNormal["time"])
            #         if metric:
            #             self.plotmakers["ARI"].addPoint(name + " metric learning", i, averageMetric["mu"])
            #             self.plotmakers["time"].addPoint(name + " metric learning", i, averageMetric["time"])
            #         tp = scipy.stats.t.ppf((1 + 0.95) / 2., runsPQ - 1)
            #         hNormal = seNormal * tp
            #         self.plotmakers["ARI"].addPoint(name + " confidence upper", i, averageNormal["mu"] + hNormal)
            #         self.plotmakers["ARI"].addPoint(name + " confidence lowe", i, averageNormal["mu"] - hNormal)
            
            if ARI:
                self.plotmakers["ARI"].viewPlot()
            # if time:
            #     self.plotmakers["time"].viewPlot()
    
    def saveResults(self): # Hier moet er pas een map gemaakt worden
        i = 0
        while True:
            if i == 0:
                resulting_path = os.path.join(self.path, self.day + "_" + self.name)
            else: 
                resulting_path = os.path.join(self.path, self.day + "_" + self.name + '_' + str(i) )
            CHECK_FOLDER = os.path.isdir(resulting_path)
            if not CHECK_FOLDER:
                os.makedirs(resulting_path)
                break
                
            else:
                i += 1
        self.path = resulting_path


    def loadResults(self):
        pass

    def clear(self):
        self.datasets = dict()
        self.plotmakers = dict()


                


# # variance test
# varianceTest(path, currentDay)
# import inquirer
# questions = [
#   inquirer.List('size',
#                 message="What size do you need?",
#                 choices=['Jumbo', 'Large', 'Standard', 'Medium', 'Small', 'Micro'],
#             ),
# ]
# answers = inquirer.prompt(questions)
# print(answers["size"])

# import os
# os.system("python experimentRunner.py")

# inp = input(f"first line{os.linesep}Second line")

