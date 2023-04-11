import abc

import numpy as np
from sklearn.neighbors import NearestNeighbors

# returns a list of labels, the labels are the indices of the repres it needs to belong to

class Rebuilder:
    @abc.abstractmethod
    def rebuild(self, repres, indices, data, indi_super=[], labelled=[]): # indi_super is de inidex in repres die overeenkomt van waar het komt, labelled heeft als lengte n + 1 om zo de laatste aan te passen naar de repres
        pass

class ClosestRebuild(Rebuilder):
    def rebuild(self, repres, indices, data, indi_super=[], labelled=[]): # werken via kNN, indices bevatten ook de represkes
        if len(labelled) == 0:
            nbrs = NearestNeighbors(n_neighbors=1).fit(data[np.array(repres)])
            _, labels = nbrs.kneighbors(data[indices])
            for i in range(len(repres)): # repres wel nog altijd hun eigen nemen (dit is alleen nodig als twee repres overeenlappen)
                labels[indi_super == repres[i]] = i # gaan er nu gewoon van uit dat het klopt
            return labels.flatten()
        labels = np.zeros(len(indices))
        for i in repres:
            selection = np.array(indi_super) == i
            labelled[-1] = i
            nbrs = NearestNeighbors(n_neighbors=1).fit(data[np.array(labelled)])
            _, lab = nbrs.kneighbors(data[selection])
            labels[selection] = lab
        for i in range(len(repres)):
                labels[indi_super == repres[i]] = i
        return labels
    
class SemiCluster(Rebuilder):
    def rebuild(self, repres, indices, data, indi_super=[], labelled=[]):
        # the inital centers
        centers = data[np.copy(repres)] # voor bestaande repres kan dit in theorie met de echte center (?)

        repers_indi = np.in1d(indices,repres)

        labels = np.zeros(len(indices))
        
        # alle repres initieel een gekend label
        for i in range(len(repres)):
            labels[indi_super == repres[i]] = i

        while True:
            # if len(labelled) == 0: geen idee of dit gaat convergeren
            nbrs = NearestNeighbors(n_neighbors=1).fit(centers)
            _, newlabels = nbrs.kneighbors(data[indices])
            newlabels = newlabels.flatten()

            # print(newlabels[repers_indi])


            if (labels == np.array(newlabels)).all(): # zijn geconvergeerd
                break
            # if not (labels[repers_indi] == np.array(indices)[repers_indi]).all(): # de repres zijn eruit geconvergeerd
            #     break
            if len(np.unique(newlabels[repers_indi])) != len(repres): # de repres moeten in verschillende clusters zitten
                break
            labels = np.copy(newlabels)


            # update the centers
            for i in range(len(centers)):
                centers[i] = data[np.array(indices)[labels == i]].mean(axis = 0)

        return labels


            # indices = np.zeros(len(indices))
            # for i in repres:
            #     selection = np.array(indi_super) == i
            #     labelled[-1] = i
            #     nbrs = NearestNeighbors(n_neighbors=1).fit(data[np.array(labelled)])
            #     _, lab = nbrs.kneighbors(data[selection])
            #     indices[selection] = lab
            # return indices


        
        