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
    def __init__(self, name: str, path, day, saveResults = False) -> None:
        self.name = name
        self.datasets = []
        self.batches = []
        self.algos = []
        self.runs = 0
        self.zippPath = path
        self.saveResults = saveResults

        
        i = 0
        while True:
            if i == 0:
                resulting_path = os.path.join(path, day + "_" + self.name)
            else: 
                resulting_path = os.path.join(path, day + "_" + self.name + '_' + str(i) )
            CHECK_FOLDER = os.path.isdir(resulting_path)
            if not CHECK_FOLDER:
                if saveResults:
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

    def saveResult(self):
        os.makedirs(self.savepath)

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

                            path_for_data = algo.preprocces(nameData, dataset_path)

                            parallel_func = functools.partial(algo.fit, nameData, path_for_data, maxQ)

                            print("Starting the run on dataset: " + nameData + " with algorithm: " + str(algo.getFileName()) , end = "\r")

                            futures = client.map(parallel_func, arguments)

                            # def test(x):
                            #     return x + 1

                            # futures = client.map(test, range(1000))
                            

                            results = client.gather(futures)

                            print("Done, total results = " + str(len(results)), end = "\r")

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
                

    def makePlot(self, maxQ, sortByAlgo = True, sortByDataset = False, seperate = False, aligned = True): # return the dataframes of this, then you can do even more things afterwards
        if aligned:
            averageARI = pd.DataFrame()
            loop = self.datasets
            # for alg in self.algos:
            for k in loop:
                mean = np.zeros(maxQ)
                i = 0
                for batch in self.batches:  
                    if batch.nameDataSet == k:
                        mean += batch.results['mu'][:maxQ]
                        i += 1
                mean = mean/i
                averageARI[k] = mean
            sortedL = []
            loop2 = [alg.getFileName() for alg in self.algos]
            for k in loop2:
                for batch in self.batches:
                    if batch.nameAlgo == k:
                        sortedL.append(averageARI[batch.nameDataSet] - batch.results['mu'][:maxQ])

            indices = np.argsort(sortedL)
            i = 0
            lenData = len(self.datasets)
            plot = pd.DataFrame()
            for k in loop2:
                print(np.average(indices[i:i + lenData], axis = 0))
                plot[k] = np.average(indices[i:i + lenData], axis = 0) 
                i += lenData

            plot.plot(title="Aligned rank", xlabel="Number of queries", ylabel="Aligned rank")
            if self.saveResults:
                plt.savefig(os.path.join(self.savepath, self.name))


        if seperate:
            frames = [pd.DataFrame() for dataset in self.datasets]
            plots = dict(zip(self.datasets, frames))
            for batch in self.batches:
                plots[batch.nameDataSet][batch.nameAlgo] = batch.results['mu'][:maxQ]

            differences = []
            for key, value in plots.items():
                value.plot(title=key, xlabel="Number of queries", ylabel="Average ARI")
                if self.saveResults:
                    plt.savefig(os.path.join(self.savepath, key))
                # differences.append(value.loc[199, "COBRAS_training_preprocessed_NCA"] - value.loc[199, "COBRAS"] )
            # plt.hist(differences)
            return plots # return to use for later use

        
        plot = pd.DataFrame()
        loop = []

        # if more than one is true => order decides which is then the default
        if sortByDataset: # dat gaat wss die seperate zijn
            loop = self.datasets
        if sortByAlgo:
            loop = [alg.getFileName() for alg in self.algos]
            sortByDataset = False
        # for alg in self.algos:
        for k in loop: # amai dit kan veel efficienter (ge doet dubbel werk ofzo)
            mean = np.zeros(maxQ)
            i = 0
            for batch in self.batches:
                if sortByDataset:
                    batchValue = batch.nameDataSet
                if sortByAlgo:
                    batchValue = batch.nameAlgo
                
                if batchValue == k:
                    mean += batch.results['mu'][:maxQ]
                    i += 1
            mean = mean/i
            plot[k] = mean



        # print(d)


        plot.plot(title=self.name, xlabel="Number of queries", ylabel="Average ARI")
        if self.saveResults:
            plt.savefig(os.path.join(self.savepath, self.name))

        return plot

    def makeZip(self):
        if self.saveResults:
            shutil.make_archive(os.path.join(self.zippPath, self.name), 'zip', self.savepath)
            shutil.move(os.path.join(self.zippPath, self.name + ".zip"), os.path.join(self.savepath, self.name + ".zip"))

