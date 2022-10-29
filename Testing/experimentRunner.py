from array import array
from audioop import mul
from math import sqrt
import math
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
import matplotlib.pyplot as plt
import shutil
import functools
from dask.distributed import Client, LocalCluster

class ExperimentRunner:
    def __init__(self, name: str, path, day) -> None:
        self.name = name
        self.datasets = []
        self.batches = []
        self.algos = []
        self.runs = 0
        self.zippPath = path

        i = 0
        while True:
            if i == 0:
                resulting_path = os.path.join(path, day + "_" + self.name)
            else: 
                resulting_path = os.path.join(path, day + "_" + self.name + '_' + str(i) )
            CHECK_FOLDER = os.path.isdir(resulting_path)
            if not CHECK_FOLDER:
                os.makedirs(resulting_path)
                break
                
            else:
                i += 1
        self.savepath = resulting_path

    def createFold(self, runsPQ, dataname): # gaan ervan uit dat het altijd 10-fold crossvalidation is
        dataset_path = Path('datasets/cobras-paper/' + dataname + '.data').absolute()
        dataset = np.loadtxt(dataset_path, delimiter=',')
        target = dataset[:, 0]
        skf = []
        for j in range(runsPQ):
            fold = StratifiedKFold(n_splits = 10, shuffle = True)
            for fold_nb, (train_indices, test_indices) in enumerate(fold.split(np.zeros(len(target)), target)):
                f = train_indices.tolist()
                f.extend(test_indices.tolist())
                skf.append(f)
        i = 1
        while True:
            resulting_path = Path('batches/' + dataname + "_crossfold_" + str(i)).absolute()
            if os.path.exists(resulting_path):
                i += 1
                
            else:
                np.savetxt(resulting_path, np.array(skf), delimiter=',')
                break

        return i # returns what number the new crossfold has

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

    def run(self, maxQ, runsPQ, crossFold = 0, save = False): # crossvalidation nog fixen
        self.runs +=1
        totalDataset = len(self.datasets)
        totalAlgos = len(self.algos)
        nbdata = 0
        

        with LocalCluster() as cluster, Client(cluster) as client:

            # loop over the datasets
            for nameData in self.datasets: # moet elk algo niet dezelfde folds hebben
                nbdata += 1

                # Load in the data
                dataset = None
                preprocessed = False
                print("                                                                      ", end = "\r")
                print("Run " + str(self.runs), end = " ")
                print("Loading data: " + str(nbdata) + "/" + str(totalDataset), end = "\r")
                dataset_path = Path('datasets/cobras-paper/' + nameData + '.data').absolute()
                dataset = np.loadtxt(dataset_path, delimiter=',')
                data = dataset[:, 1:]
                target = dataset[:, 0]

                # loop over the algorithms
                nbalg = 0
                for algo in self.algos: # schrijf dit allemaal in functies
                    nbalg += 1

                    # create the batch
                    batch = Batch(nameData, algo.getFileName(), maxQ, runsPQ, crossFold)

                    # see if you already have the results of the run
                    if batch.chechIfRunned():
                        self.batches.append(batch)
                        continue
                    
                    # start the run
                    else: 

                        if crossFold:
                            # en vanaf hier moet het parallel worden
                            # execute crossfold validation
                            average = {"S1": np.zeros(maxQ), "S2": np.zeros(maxQ), "times": np.zeros(maxQ)}
                            resulting_path = Path('batches/' + nameData + "_crossfold_" + str(crossFold)).absolute()
                            folds = np.loadtxt(resulting_path, delimiter=',', dtype=int)

                            arguments = []
                            test = []

                            size = len(folds[0])
                            amount = math.ceil(0.1 * size)

                            for fold in folds:
                                arguments.append(fold[0:-amount])
                                test.append(fold[-amount:])

                            newData = algo.preprocces(dataName= nameData, data = data, target = target)

                            parallel_func = functools.partial(algo.fit, nameData, np.copy(newData), np.copy(target), maxQ)

                            print("yeet")

                            futures = client.map(parallel_func, arguments)

                            # def test(x):
                            #     return x + 1

                            # futures = client.map(test, range(1000))
                            
                            print("start the waiting")

                            results = client.gather(futures)

                            print("done waiting")

                            for i in range(len(results)):
                                all_clusters, runtimes = results[i]
                                if len(all_clusters) < maxQ:
                                    diff = maxQ - len(all_clusters)
                                    for ex in range(diff):
                                        all_clusters.append(all_clusters[-1])
                                        runtimes.append(runtimes[-1])
                                    
                                IRA = np.array([adjusted_rand_score(target[test[i]], np.array(clustering)[test[i]]) for clustering in all_clusters])
                                average["S1"] += IRA
                                average["S2"] += IRA**2
                                average["times"] += np.array(runtimes)

                            batch.results["mu"] = average["S1"]/(runsPQ*10)
                            batch.results["times"] = average["times"]/(runsPQ*10)
                            seNormal = np.sqrt(((runsPQ*10)*average["S2"] - average["S1"]**2)/((runsPQ*10)*((runsPQ*10)-1)))
                            tp = scipy.stats.t.ppf((1 + 0.95) / 2., (runsPQ*10) - 1)
                            batch.results["hNormal"] = seNormal * tp


                            
                            # for foldNb in range(len(folds)): # momenteel doet runsPQ hier niks
                            #     # print the progress
                            #     # def prf():
                            #     fold = folds[foldNb]
                            #     print("                                                                                         ", end="\r" )
                            #     print("Run " + str(self.runs), end = " ")
                            #     print("dataset: " + str(nbdata) + "/" + str(totalDataset), end = " ")
                            #     print("algo: " + str(nbalg) + "/" + str(totalAlgos), end = " ")
                            #     print("fold: " + str(foldNb + 1) + "/" + str(100), end=" ")

                            #     size = len(fold)
                            #     amount = math.ceil(0.1 * size)

                            #     train_indices = fold[0:-amount]
                            #     test_indices = fold[-amount:]

                            #     all_clusters, runtimes = algo.fit(nameData, np.copy(data), np.copy(target), maxQ, trainingset=train_indices, prf = None)  #prf = ExperimentLogger(prf)
                            #     if len(all_clusters) < maxQ:
                            #         diff = maxQ - len(all_clusters)
                            #         for ex in range(diff):
                            #             all_clusters.append(all_clusters[-1])
                            #             runtimes.append(runtimes[-1])
                                    
                            #     IRA = np.array([adjusted_rand_score(target[test_indices], np.array(clustering)[test_indices]) for clustering in all_clusters])
                            #     average["S1"] += IRA
                            #     average["S2"] += IRA**2
                            #     average["times"] += np.array(runtimes) # TODO, omzetten naar f strings

                            # batch.results["mu"] = average["S1"]/(runsPQ*10)
                            # batch.results["times"] = average["times"]/(runsPQ*10)
                            # seNormal = np.sqrt(((runsPQ*10)*average["S2"] - average["S1"]**2)/((runsPQ*10)*((runsPQ*10)-1)))
                            # tp = scipy.stats.t.ppf((1 + 0.95) / 2., (runsPQ*10) - 1)
                            # batch.results["hNormal"] = seNormal * tp
                            

                                    
                        else: # dees moeten we nog is bekijken
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

                                all_clusters, runtimes = algo.fit(nameData,np.copy(data), np.copy(target), maxQ, prf())
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
                

    def makePlot(self, maxQ, sortByAlgo = True, sortByDataset = False, seperate = False): # return the dataframes of this, then you can do even more things afterwards
        if seperate:
            frames = [pd.DataFrame() for dataset in self.datasets]
            plots = dict(zip(self.datasets, frames))
            for batch in self.batches:
                plots[batch.nameDataSet][batch.nameAlgo] = batch.results['mu']

            differences = []
            for key, value in plots.items():
                value.plot(title=key, xlabel="Number of queries", ylabel="Average ARI")
                plt.savefig(os.path.join(self.savepath, key))
                differences.append(value.loc[199, "COBRAS_training_preprocessed_NCA"] - value.loc[199, "COBRAS"] )

            # plt.hist(differences)
            shutil.make_archive(os.path.join(self.zippPath, self.name), 'zip', self.savepath)
            shutil.move(os.path.join(self.zippPath, self.name + ".zip"), os.path.join(self.savepath, self.name + ".zip"))
            return 
        plot = pd.DataFrame()
        loop = []

        # if more than one is true => order decides which is then the default
        if sortByDataset:
            loop = self.datasets
        if sortByAlgo:
            loop = [alg.getFileName() for alg in self.algos]
            sortByDataset = False
        # for alg in self.algos:
        for k in loop:
            mean = np.zeros(maxQ)
            i = 0
            for batch in self.batches:
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

