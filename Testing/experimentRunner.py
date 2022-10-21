from array import array
from audioop import mul
from math import sqrt
from pathlib import Path
import os
from statistics import mean
from sklearn.metrics import adjusted_rand_score
from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.noisy_labelquerier import ProbabilisticNoisyQuerier
from noise_robust_cobras.querier.labelquerier import LabelQuerier
import noise_robust_cobras.metric_learning.metriclearning_algorithms 
import numpy as np 
import scipy
from metric_learn import NCA
from batch import Batch
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
import sklearn as sk
from experimentLogger import ExperimentLogger

class ExperimentRunner:
    def __init__(self, name: str, path, day) -> None:
        self.name = name
        self.datasets = []
        self.batches = []
        self.algos = []
        self.runs = 0

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
    def addAlgo(self, algos):
        if len(algos) == 0:
            print("load all algo's")
        self.algos = algos

    def loadDataSets(self, datasets:array): 
        if len(datasets) == 0:
            path = Path('datasets/cobras-paper/').absolute()
            dir_list = os.listdir(path)
            for i in dir_list:
                self.datasets.append(i[:len(i) - 5])
        else:
            path = Path('datasets/cobras-paper/').absolute()
            for set in datasets:
                data_path = os.path.join(path, set + ".data")
                if not os.path.exists(data_path):
                    datasets.remove(set)
                    continue

            self.datasets = datasets

    def clear(self):
        self.datasets = []
        self.batches = []
        self.algos = []

    def SaveBatches(self):
        for batch in self.batches:
            batch.saveResults()

        self.batches = []

    def run(self, maxQ, runsPQ, crossFold = False, metricPreprocessing = False, save = False): # crossvalidation nog fixen
        self.runs +=1
        totalDataset = len(self.datasets)
        totalAlgos = len(self.algos)
        nbdata = 0

        # loop over the datasets
        for nameData in self.datasets: # moet elk algo niet dezelfde folds hebben
            nbdata += 1

            # Load in the data
            dataset = None
            preprocessed = False
            print("                                                                      ", end = "\r")
            print("Run " + str(self.runs), end = " ")
            print("Loading data: " + str(nbdata) + "/" + str(totalDataset), end = "\r")
            if (metricPreprocessing):
                path_pre = Path('batches/' + nameData + "_" + "preprocessed_" + type(metricPreprocessing).__name__).absolute()  
                if os.path.exists(path_pre):
                    dataset = np.loadtxt(path_pre, delimiter=',')
                    preprocessed = True

                else:
                    dataset_path = Path('datasets/cobras-paper/' + nameData + '.data').absolute()
                    dataset = np.loadtxt(dataset_path, delimiter=',')
                    
            else:
                dataset_path = Path('datasets/cobras-paper/' + nameData + '.data').absolute()
                dataset = np.loadtxt(dataset_path, delimiter=',')
                preprocessed = True 
            data = dataset[:, 1:]
            target = dataset[:, 0]

            # create the crossfolds
            skf = []
            if crossFold:
                print("                                                                      ", end = "\r")
                print("Run " + str(self.runs), end = " ")
                print("Creating crossfolds: " + str(nbdata) + "/" + str(totalDataset), end = "\r")
                for j in range(runsPQ):
                    skf.append(StratifiedKFold(n_splits = 10, shuffle = True)) 

            # loop over the algorithms
            nbalg = 0
            for algo in self.algos: # schrijf dit allemaal in functies
                nbalg += 1

                # create the batch
                batch = Batch(nameData, algo.getFileName(), maxQ, runsPQ, crossFold, metricPreprocessing)

                # see if you already have the results of the run
                if batch.chechIfRunned():
                    self.batches.append(batch)
                    continue
                
                # start the run
                else: 

                    # if not already preprocessed, do it
                    if (not preprocessed):
                        path_pre = Path('batches/' + nameData + "_" + "preprocessed_" + type(metricPreprocessing).__name__).absolute()    
                        print("                                                                      ", end = "\r")
                        print("Run " + str(self.runs), end = " ")
                        print("Metric learning om full data: " + str(nbdata) + "/" + str(totalDataset), end = "\r")
                        preprocessor = sk.base.clone(metricPreprocessing, safe=True) # need a new empty model
                        preprocessor.fit(np.copy(data), np.copy(target))
                        data = preprocessor.transform(data)
                        preprocessed = True
                        # save the processed data for later use
                        np.savetxt(path_pre, np.column_stack((target,data)), delimiter=',')

                    if crossFold:
                        # execute crossfold validation
                        average = {"S1": np.zeros(maxQ), "S2": np.zeros(maxQ), "times": np.zeros(maxQ)}
                        for j in range(runsPQ):
                            for fold_nb, (train_indices, test_indices) in enumerate(skf[j].split(np.zeros(len(target)), target)):
                                # print the progress
                                def prf():
                                    print("                                                                                         ", end="\r" )
                                    print("Run " + str(self.runs), end = " ")
                                    print("dataset: " + str(nbdata) + "/" + str(totalDataset), end = " ")
                                    print("algo: " + str(nbalg) + "/" + str(totalAlgos), end = " ")
                                    print("foldrun: " + str(j + 1) + "/" + str(runsPQ), end = " ")
                                    print("fold: " + str(fold_nb + 1) + "/" + str(10), end=" ")

                                all_clusters, runtimes = algo.fit(np.copy(data), np.copy(target), maxQ, trainingset=train_indices, prf = ExperimentLogger(prf))
                                if len(all_clusters) < maxQ:
                                    diff = maxQ - len(all_clusters)
                                    for ex in range(diff):
                                        all_clusters.append(all_clusters[-1])
                                        runtimes.append(runtimes[-1])
                                    
                                IRA = np.array([adjusted_rand_score(target[test_indices], np.array(clustering)[test_indices]) for clustering in all_clusters])
                                average["S1"] += IRA
                                average["S2"] += IRA**2
                                average["times"] += np.array(runtimes) # TODO, omzetten naar f strings

                        batch.results["mu"] = average["S1"]/(runsPQ*10)
                        batch.results["times"] = average["times"]/(runsPQ*10)
                        seNormal = np.sqrt(((runsPQ*10)*average["S2"] - average["S1"]**2)/((runsPQ*10)*((runsPQ*10)-1)))
                        tp = scipy.stats.t.ppf((1 + 0.95) / 2., (runsPQ*10) - 1)
                        batch.results["hNormal"] = seNormal * tp
                        

                                
                    else:
                        # no cross-fold validation
                        average = {"S1": np.zeros(maxQ), "S2": np.zeros(maxQ), "times": np.zeros(maxQ)}
                        for j in range(runsPQ):

                            # print the progress
                            def prf():
                                print("                                                                  ", end="\r" )
                                print("Run " + str(self.runs), end = " ")
                                print(f"{((nbdata - 1)*runsPQ*totalAlgos + runsPQ*(nbalg - 1) + j + 1)/(runsPQ*totalAlgos*totalDataset)*100:.1f} %", end=" ")
                                print("dataset: " + str(nbdata) + "/" + str(totalDataset), end = " ")
                                print("algo: " + str(nbalg) + "/" + str(totalAlgos), end = " ")
                                print(f"{(j + 1)/(runsPQ)*100:.1f} %", end=" ")

                            all_clusters, runtimes = algo.fit(np.copy(data), np.copy(target), maxQ, prf())
                            IRA = np.array([adjusted_rand_score(target, clustering) for clustering in all_clusters])
                            average["S1"] += IRA
                            average["S2"] += IRA**2
                            average["times"] += np.array(runtimes)
                        batch.results["mu"] = average["S1"]/runsPQ
                        batch.results["times"] = average["times"]/runsPQ
                        seNormal = np.sqrt((runsPQ*average["S2"] - average["S1"]**2)/(runsPQ*(runsPQ-1)))
                        tp = scipy.stats.t.ppf((1 + 0.95) / 2., runsPQ - 1)
                        batch.results["hNormal"] = seNormal * tp
                if save:
                    batch.saveResults()
                self.batches.append(batch)
                

    def makePlot(self, maxQ, sortByAlgo = True, sortByPreprocessing = False, sortByDataset = False):
        plot = pd.DataFrame()
        loop = []

        # if more than one is true => order decides which is then the default
        if sortByPreprocessing:
            loop = ["Preprocessed with metric learning", "Not preprocessed"]
        if sortByDataset:
            loop = self.datasets
            sortByPreprocessing = False
        if sortByAlgo:
            loop = [alg.getFileName() for alg in self.algos]
            sortByPreprocessing = False
            sortByDataset = False
        # for alg in self.algos:
        for k in loop:
            mean = np.zeros(maxQ)
            i = 0
            for batch in self.batches:
                batchValue =  None
                if sortByPreprocessing:
                    if batch.metricPreprocessing:
                        batchValue = loop[0]
                    else:
                        batchValue = loop[1]
                if sortByDataset:
                    batchValue = batch.nameDataSet
                if sortByAlgo:
                    batchValue = batch.nameAlgo
                
                if batchValue == k:
                    mean += batch.results['mu']
                    i += 1
            mean = mean/i
            plot[k] = mean


        # print(d)


        plot.plot(title=self.name, xlabel="Number of queries", ylabel="Average ARI")






        # if (all): 
        #     path = Path('datasets/cobras-paper/').absolute()
        #     dir_list = os.listdir(path)
        #     for i in dir_list:
        #         p = os.path.join(path, i)
        #         dataset = np.loadtxt(p, delimiter=',')
        #         data = dataset[:, 1:]
        #         target = dataset[:, 0]
        #     return False

        # signal = True
        # if datasets is None:
        #     inp = None
        #     while True:
        #         path = Path('datasets/cobras-paper/').absolute()
        #         dir_list = os.listdir(path)
        #         string = ""
        #         for i in dir_list:
        #             string = string + i[:len(string) - 5] + " | " 

        #         inp = input(string)
        #         if not inp:
        #             return signal
        #         dataset_path = Path('datasets/cobras-paper/' + inp + ".data").absolute() # brow er wordt nergens gecheched of die al bestaat
        #         if (metricPreprocessing):
        #             inp = inp + " " + "metricPreprocessing"
        #         print(inp)
        #         samples = self.datasets.get(inp, None)
        #         if samples is not None:
        #             print("bestaat al kut")
        #             continue
        #         signal = False
        #         dataset = np.loadtxt(dataset_path, delimiter=',')
        #         data = dataset[:, 1:]
        #         target = dataset[:, 0]
        #         if (metricPreprocessing):
        #             l = NCA(max_iter=1000)
        #             l.fit(np.copy(data), np.copy(target))
        #             data = l.transform(data)


        #         self.datasets[inp] = {
        #                                 "data": data,
        #                                 "target": target,
        #                                 }

    # def run(self, ARI = True, time = False, plot = False, 
    #         maxQ = 100, runsPQ = 1, CI = False): # returns de COBRAS indien een run
    #         self.plotmakers["ARI"] = IRAPlot("ARI plot")
    #         # self.plotmakers["time"] = IRAPlot("time plot")

    #         k = 0
    #         print(len(self.algos))
    #         for name_algo,samples in self.algos.items():
    #             k += 1
    #             for name,samples in self.datasets.items():
    #                 average = {"S1": np.zeros(maxQ), "S2": np.zeros(maxQ), "mu": np.zeros(maxQ)}
    #                 for j in range(runsPQ):
    #                     querier = LabelQuerier(None, samples["target"], maxQ)
    #                     clusterer = COBRAS(correct_noise=False)
    #                     all_clusters, runtimes, *_ = clusterer.fit(samples["data"], -1, None, querier)
    #                     # best_clustering = all_clusters[-1]
    #                     # runtime = runtimes[-1]
    #                     IRA = np.array([adjusted_rand_score(samples["target"], clustering) for clustering in all_clusters])
                        
    #                     average["S1"] += IRA
    #                     average["S2"] += IRA**2
        
    #                     print(f"{((k-1)* runsPQ + j + 1)/(runsPQ*len(self.algos))*100:.1f} %", end="\r")

    #                 average["mu"] = average["S1"]/runsPQ
    #                 seNormal = np.sqrt((runsPQ*average["S2"] - average["S1"]**2)/(runsPQ*(runsPQ-1)))
    #                 tp = scipy.stats.t.ppf((1 + 0.95) / 2., runsPQ - 1)
    #                 hNormal = seNormal * tp

    #                 self.plotmakers["ARI"].addPoint(str(name_algo) + " " + name + " no metric learning", average["mu"])
    #                 # self.plotmakers["ARI"].addPoint(str(name_algo) + " " + name + " upper", average["mu"] + hNormal)
    #                 # self.plotmakers["ARI"].addPoint(str(name_algo) + " " + name + " lower", average["mu"] - hNormal)

    #         # for name,samples in self.datasets.items():
    #         #     for i in range(maxQ):
    #         #         averageNormal = {"S1": 0, "S2": 0, "time": 0, "mu": 0}
    #         #         averageMetric = {"S1": 0, "S2": 0, "time": 0, "mu": 0}
    #         #         for j in range(runsPQ):
    #         #             if normal:
    #         #                 querier = LabelQuerier(None, samples["target"], i + 1)
    #         #                 clusterer = COBRAS(correct_noise=False)
    #         #                 all_clusters, runtimes, *_ = clusterer.fit(samples["data"], -1, None, querier)
    #         #                 best_clustering = all_clusters[-1]
    #         #                 runtime = runtimes[-1]
    #         #                 averageNormal["S1"] += adjusted_rand_score(samples["target"], best_clustering)
    #         #                 averageNormal["S2"] += adjusted_rand_score(samples["target"], best_clustering)**2
    #         #                 averageNormal["time"] += (runtime - averageNormal["time"])/(j + 1)
    #         #                 print(f"{((i*runsPQ + j) * maal + 1)/(totalRuns)*100:.1f} %", end="\r")
    #         #             if metric:
    #         #                 querier = LabelQuerier(None, samples["target"], i + 1)
    #         #                 clusterer = COBRAS(correct_noise=False, metric_algo=SemiSupervisedMetric())
    #         #                 all_clusters, runtimes, *_ = clusterer.fit(samples["data"], -1, None, querier)
    #         #                 best_clustering = all_clusters[-1]
    #         #                 runtime = runtimes[-1]
    #         #                 averageMetric["S1"] += adjusted_rand_score(samples["target"], best_clustering)
    #         #                 averageMetric["S2"] += adjusted_rand_score(samples["target"], best_clustering)**2
    #         #                 averageMetric["time"] += (runtime - averageMetric["time"])/(j + 1)
    #         #                 print(f"{((i*runsPQ + j) * maal + 2)/(totalRuns)*100:.1f} %", end="\r")
    #         #         averageNormal["mu"] = averageNormal["S1"]/runsPQ
    #         #         averageMetric["mu"] = averageMetric["S1"]/runsPQ
    #         #         seNormal = sqrt((runsPQ*averageNormal["S2"] - averageNormal["S1"]**2)/(runsPQ*(runsPQ-1)))
    #         #         seMetric = 0 # nog ff niet nodig
    #         #         if normal:
    #         #             self.plotmakers["ARI"].addPoint(name + " no metric learning", i, averageNormal["mu"])
    #         #             self.plotmakers["time"].addPoint(name + " no metric learning", i, averageNormal["time"])
    #         #         if metric:
    #         #             self.plotmakers["ARI"].addPoint(name + " metric learning", i, averageMetric["mu"])
    #         #             self.plotmakers["time"].addPoint(name + " metric learning", i, averageMetric["time"])
    #         #         tp = scipy.stats.t.ppf((1 + 0.95) / 2., runsPQ - 1)
    #         #         hNormal = seNormal * tp
    #         #         self.plotmakers["ARI"].addPoint(name + " confidence upper", i, averageNormal["mu"] + hNormal)
    #         #         self.plotmakers["ARI"].addPoint(name + " confidence lowe", i, averageNormal["mu"] - hNormal)
            
    #         if ARI:
    #             self.plotmakers["ARI"].viewPlot()
    #         # if time:
    #         #     self.plotmakers["time"].viewPlot()
    
    # def saveResults(self): # Hier moet er pas een map gemaakt worden
    #     i = 0
    #     while True:
    #         if i == 0:
    #             resulting_path = os.path.join(self.path, self.day + "_" + self.name)
    #         else: 
    #             resulting_path = os.path.join(self.path, self.day + "_" + self.name + '_' + str(i) )
    #         CHECK_FOLDER = os.path.isdir(resulting_path)
    #         if not CHECK_FOLDER:
    #             os.makedirs(resulting_path)
    #             break
                
    #         else:
    #             i += 1
    #     self.path = resulting_path


    # def loadResults(self):
    #     pass


                


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

