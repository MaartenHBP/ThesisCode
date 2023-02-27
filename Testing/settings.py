import numpy as np
from pathlib import Path
from abc import abstractmethod
from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.labelquerier import LabelQuerier
from noise_robust_cobras.metric_learning.metriclearning_algorithms import * 
from noise_robust_cobras.metric_learning.metricLearners import * 
from metric_learn import *
from pathlib import Path
import os
import pandas as pd
from sklearn.cluster import KMeans
import math
from sklearn.cluster import SpectralClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

class metricSettings:
    def __init__(self, metric, data, parameters, name, typeMetric, k = 2):
        self.name = name
        self.typeMetric = typeMetric
        self.metric = metric

        # path = Path(f'{datasetpath}.data').absolute()
        # data= np.loadtxt(path, delimiter=',')
        # self.dataset = data

        self.data = data

        self.parameters = parameters
        self.constraints = None
        self.pairs = None
        self.transformed = None
        self.clustOriginal = None   
        self.clustTransformed = None
        self.k = k

        self.clustering = None

    def changeMertic(self, metric, parameters, typeMetric):
        self.metric = metric
        self.typeMetric = typeMetric
        self.parameters = parameters
        self.transformed = None
        self.clustOriginal = None   
        self.clustTransformed = None

    def changeData(self, data):
        self.data = data
        self.transformed = None
        self.clustOriginal = None   
        self.clustTransformed = None

    def changeName(self, name):
        self.name = name
        return self

    def changeK(self, newK):
        self.k = newK

    def copy(self, name):
        new = metricSettings(self.metric, self.data, dict(self.parameters), name, self.typeMetric, self.k)
        new.constraints = np.copy(self.constraints) # zeer nuttig om testen uit te voeren
        new.pairs = np.copy(self.pairs)
        return new

    def newConstraints(self, data, nb_links):
        if (self.typeMetric == "supervised"):
            print("supervised constraints")
            self.constraints = np.copy(data[:,0])

        else:
            AMOUNT_OF_LINKS_WANTED = nb_links # also make this a parameter
            print("semisupervised constraints")
            labels = data[:,0] # mss hier ook nog met transformed werken
            nbLinks = labels.shape[0]
            maxQ = math.floor(AMOUNT_OF_LINKS_WANTED)
            querier = LabelQuerier(None, labels, maxQ)
            clusterer = COBRAS(correct_noise=False, seed=42)
            all_clusters, runtimes, superinstances, clusterIteration, transformations, ml, cl = clusterer.fit(data[:,1:], -1, None, querier)
            pairs = np.vstack((ml,cl))
            constrains = np.full(len(ml) + len(cl), 1)
            constrains[len(ml):] = np.full(len(cl), -1)
            self.pairs, self.constraints = expand(pairs, constrains)

    def learnMetric(self, data, onOrig = True):
        # self.transformed = np.copy(data)
        if self.constraints is None:
            print("Need constraints")
            return
        if onOrig:
            print("learning metric on original data")
            if (self.typeMetric == "supervised"):
                metric = eval(self.metric)(**self.parameters)
                self.transformed = metric.fit(np.copy(data), self.constraints).transform(np.copy(data))
            else:
                metric = eval(self.metric)(preprocessor=np.copy(data),**self.parameters)
                self.transformed = metric.fit(self.pairs, self.constraints).transform(np.copy(data))
            
        else:
            if not self.transformed is None:
                print("learning metric on transformed data")
                if (self.typeMetric == "supervised"):
                    metric = eval(self.metric)(**self.parameters)
                    self.transformed = metric.fit(np.copy(self.transformed), self.constraints).transform(np.copy(self.transformed))
                else:
                    metric = eval(self.metric)(preprocessor=np.copy(data),**self.parameters)
                    self.transformed = metric.fit(self.pairs, self.constraints).transform(np.copy(self.transformed))

            else:
                print("No transformed data available") # hier had je met errors kunnen werken thooo
            
        self.clustOriginal = None   
        self.clustTransformed = None

    #  dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y), werken via een callable voor kernel zeverrrrrrr
    def executeKMedoids(self, data):
        mlconstraints = self.pairs[self.constraints == 1]
        clconstraints = self.pairs[self.constraints == -1]

        distances = pairwise_distances(data, metric='euclidean')
        nbrs = NearestNeighbors(radius=0.1).fit(data)
        shortest, indices = nbrs.radius_neighbors(data)

        kmedoids = KMedoids(n_clusters=self.k, metric='precomputed').fit(distances)
        self.clustTransformed = kmedoids.labels_

        ####################################################
        blobs = []
        seen_indices = [] # deze zitten dus in ML blobs
        # make blobs
        for ml in mlconstraints:
            ind1 = ml[0]
            ind2 = ml[1]
            blob1 = []
            blob2 = []
            if ind1 in seen_indices:
                for blob in blobs:
                    if ind1 in blob:
                        blob1 = blob
                        break
            if ind2 in seen_indices:
                for blob in blobs:
                    if ind2 in blob:
                        blob2 = blob
                        break

            if len(blob1) > 0 and len(blob2) > 0:
                blob1.extend(blob2)
                blobs.remove(blob2)
                continue
            if len(blob1) > 0:
                blob1.append(ind2)
                seen_indices.append(ind2)
                continue
            if len(blob2) > 0:
                blob2.append(ind1)
                seen_indices.append(ind1)
                continue
            blobs.append([ind1, ind2])
            seen_indices.extend([ind1, ind2])

        # NN, alle punten aan een blob toevoegen

        for i in range(len(data)):
            if not i in seen_indices:
                pass

        # fullblobs = []
        for blob in blobs:
            # print(distances[blob, :][:, blob])
            # indi = list(set(np.array(indices)[blob].flatten()))
            indi = []
            for e in blob:
                indi.extend(indices[e])

            indi = list(set(indi))
            # fullblobs.append(list(set(indi))) # dubbele er ook uithalen, is in theorie mogelijk

            distances[np.ix_(blob,blob)] = 0
            distances[np.ix_(indi,indi)] = 0
            # print(distances[blob, :][:, blob])

        for cl in clconstraints:
            ind1 = cl[0]
            ind2 = cl[1]
            left = [ind1]
            right = [ind2]

            # fix the fullblobs:
            # found = []
            # for blob in fullblobs:
            #     if ind1 in blob:
            #         found.append(ind1)
            #     if ind2 in blob:
            #         found.append(ind2)
            #     if len(found) == 1:
            #         found = []
            #         continue
            #     if len(found) == 2: # alle 2 opeens in dezelfde blob
            #         print("hier")
            #         if (ind1 in seen_indices): # ervan uitgaan dat er geen fouten gemaakt zijn
            #             blob.remove(ind2)
            #             found = [] # een punt kan in meerdere blobs zijn terechtgekomen in theorie
            #             continue
            #         if (ind2 in seen_indices):
            #             blob.remove(ind1)
            #             found = []
            #             continue
            #         blob.remove(ind1) # beide gewoon removen
            #         blob.remove(ind2)
            #         found = []
                    


            if ind1 in seen_indices:
                for blob in blobs:
                    if ind1 in blob:
                        left = blob
                        break

            if ind2 in seen_indices:
                for blob in blobs:
                    if ind2 in blob:
                        right = blob
                        break

            # indileft = np.array(indices)[left].flatten()
            # indiright = np.array(indices)[right].flatten()
            
            # print(distances[left, :][:, right])
            distances[np.ix_(left,right)] = distances.max()
            distances[np.ix_(right, left)] = distances.max()
            print(distances[left, :][:, right])

        # for blob in fullblobs:
        #     # print(distances[blob, :][:, blob])

        #     distances[np.ix_(blob,blob)] = 0

        


        ####################################################

        
        
       
        # print(distances)
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print(distances[mlconstraints[:, 0], mlconstraints[:, 1]])
        # distances[mlconstraints[:, 0], mlconstraints[:, 1]] = 0
        # distances[mlconstraints[:, 0], mlconstraints[:, 1]] = 0
        # print(distances[mlconstraints[:, 0], mlconstraints[:, 1]])
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print( distances[clconstraints[:, 0], clconstraints[:, 1]])
        # distances[clconstraints[:, 0], clconstraints[:, 1]] = distances.max()
        # distances[clconstraints[:, 1], clconstraints[:, 0]] = distances.max()
        # print( distances[clconstraints[:, 0], clconstraints[:, 1]])
        # print(distances)

        # for ml in mlconstraints: # ge moogt hier niet hunzelf en de andere nemen ofc,maar is nu nog geen probleem
        #     leftShortest = indices[ml[0]]
        #     print(leftShortest)

        #     rightShortest = indices[ml[1]]
        #     print(rightShortest)

        #     transform = data[ml[0]] -  data[ml[1]]

            

        #     # ingewikkelde shit nodig, ma in se gaan we gewoon verplaatsen van een kant


        #     # leftMax = shortest[ml[0]][9]
        #     # rightMax = shortest[ml[1]][9]


        kmedoids = KMedoids(n_clusters=self.k, metric='precomputed').fit(distances)
        self.clustOriginal = kmedoids.labels_
        print("executed kmedoids")

        


    def executeClustering(self, data):
        kmeans = KMeans(self.k).fit(data)
        # kmeans = SpectralClustering(n_clusters=2,
        # eigen_solver="arpack", affinity='nearest_neighbors').fit(data)
        # kmeans.fit(data)
        self.clustOriginal = kmeans.labels_
        if not self.transformed is None:
            kmeans = KMeans(self.k)
            kmeans.fit(self.transformed)
            self.clustTransformed = kmeans.labels_

    def exectuteSpectral(self, data): # ff snel gemaakt voor te testen TODO
        kmeans = SpectralClustering(n_clusters=2, eigen_solver="arpack", affinity='nearest_neighbors', n_neighbors= 10).fit(data)
        # kmeans = SpectralClustering(n_clusters=2,
        # eigen_solver="arpack", affinity='nearest_neighbors').fit(data)
        # kmeans.fit(data)
        self.clustOriginal = kmeans.labels_
        if not self.transformed is None:
            sp = SpectralClustering(n_clusters=2, eigen_solver="arpack", affinity='nearest_neighbors', n_neighbors= 10).fit(self.transformed)
            self.clustTransformed = sp.labels_

    def executeSpectralSpecial(self, data):
        kmeans = SpectralClustering(n_clusters=2, eigen_solver="arpack", affinity='nearest_neighbors', n_neighbors= 10).fit(data)
        aff = kmeans.affinity_matrix_
        yc = np.copy(self.constraints)
        yc[yc==-1] = 0
        aff[self.pairs[:,0], self.pairs[:,1]] = yc
        aff[self.pairs[:,1], self.pairs[:,0]] = yc
        kmeans = SpectralClustering(n_clusters=2, eigen_solver="arpack", affinity='precomputed').fit(aff)
        # kmeans = SpectralClustering(n_clusters=2,
        # eigen_solver="arpack", affinity='nearest_neighbors').fit(data)
        # kmeans.fit(data)
        self.clustOriginal = kmeans.labels_
        if not self.transformed is None:
            kmeans = SpectralClustering(n_clusters=2, eigen_solver="arpack", affinity='nearest_neighbors', n_neighbors= 10).fit(self.transformed)
            aff = kmeans.affinity_matrix_
            yc = np.copy(self.constraints)
            yc[yc==-1] = 0
            aff[self.pairs[:,0], self.pairs[:,1]] = yc
            aff[self.pairs[:,1], self.pairs[:,0]] = yc
            kmeans = SpectralClustering(n_clusters=2, eigen_solver="arpack", affinity='precomputed').fit(aff)
            self.clustTransformed = kmeans.labels_


    # def executeCOBRAS(self, data, onOrig = True):
    #     if self.transformed and not onOrig:
    #         labels = self.transformed[:,0]
    #         querier2 = LabelQuerier(None, labels, 200)
    #         clusterer = COBRAS(correct_noise=False, logExtraInfo=True)
    #         all_clusters, runtimes, superinstances, clusterIteration, *_ = clusterer.fit(self.transformed[:,1:], -1, None, querier2)
    #         self.cobrasTransformed = [superinstances, clusterIteration]

    #     if onOrig:
    #         labels = data[:,0]
    #         querier2 = LabelQuerier(None, labels, 200)
    #         clusterer = COBRAS(correct_noise=False, logExtraInfo=True)
    #         all_clusters, runtimes, superinstances, clusterIteration, *_ = clusterer.fit(data[:,1:], -1, None, querier2)
    #         self.cobrasOriginal = [superinstances, clusterIteration]

        # animation should be saved here

def expand(pairs, y):
    newpairs = []
    newy = []
    blobs = createBlobs(pairs[y == 1])
    for blob in blobs:
        new = list(itertools.combinations(blob, 2))
        newpairs.extend(new)
        newy.extend([1]*len(new))

    for cl in pairs[y == -1]:
        left = min(cl)
        right = max(cl)
        if ([left, right] in newpairs):
            continue
        leftblob = [left]
        rightblob = [right]

        found = 0
        for blob in blobs:
            if left in blob:
                leftblob = blob
                found += 1
            if right in blob:
                rightblob = blob
                found += 1
            if found == 2:
                break
        
        newcl = np.transpose([np.tile(leftblob, len(rightblob)), np.repeat(rightblob, len(leftblob))])

        newpairs.extend(newcl.tolist())
        newy.extend([-1]*len(newcl))

    return np.array(newpairs), np.array(newy)


def createBlobs(must_links):
    blobs = []
    seen_indices = [] # deze zitten dus in ML blobs
    for ml in must_links:
        ind1 = ml[0]
        ind2 = ml[1]
        blob1 = []
        blob2 = []
        if ind1 in seen_indices:
            for blob in blobs:
                if ind1 in blob:
                    blob1 = blob
                    break
        if ind2 in seen_indices:
            for blob in blobs:
                if ind2 in blob:
                    blob2 = blob
                    break

        if len(blob1) > 0 and len(blob2) > 0:
            blob1.extend(blob2)
            blobs.remove(blob2)
            continue
        if len(blob1) > 0:
            blob1.append(ind2)
            seen_indices.append(ind2)
            continue
        if len(blob2) > 0:
            blob2.append(ind1)
            seen_indices.append(ind1)
            continue
        blobs.append([ind1, ind2])
        seen_indices.extend([ind1, ind2])

    return blobs

