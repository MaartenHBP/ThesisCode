import abc

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

# returns a list of labels, the labels are the indices of the repres it needs to belong to

class Rebuilder:
    @abc.abstractmethod 
    def rebuild(self, repres, indices, data, represLabels):  # indi_super zijn de punten in de indices die een representatieve voorstellen
        pass

class ClosestRebuild(Rebuilder):
    def rebuild(self, repres, indices, data, represLabels): # werken via kNN, indices bevatten ook de represkes, doen momenteel niks met closest
        # if len(labelled) == 0:
        nbrs = NearestNeighbors(n_neighbors=1).fit(data[np.array(repres)])
        _, labels = nbrs.kneighbors(data[indices])
        for i in range(len(repres)): # repres wel nog altijd hun eigen nemen (dit is alleen nodig als twee repres overeenlappen)
            labels[np.array(indices) == repres[i]] = i # gaan er nu gewoon van uit dat het klopt
        return labels.flatten()
        # labels = np.zeros(len(indices))
        # for i in repres:
        #     selection = np.array(indi_super) == i
        #     labelled[-1] = i
        #     nbrs = NearestNeighbors(n_neighbors=1).fit(data[np.array(labelled)])
        #     _, lab = nbrs.kneighbors(data[selection])
        #     labels[selection] = lab
        # for i in range(len(repres)):
        #         labels[indi_super == repres[i]] = i
        # return labels
    



            # indices = np.zeros(len(indices))
            # for i in repres:
            #     selection = np.array(indi_super) == i
            #     labelled[-1] = i
            #     nbrs = NearestNeighbors(n_neighbors=1).fit(data[np.array(labelled)])
            #     _, lab = nbrs.kneighbors(data[selection])
            #     indices[selection] = lab
            # return indices

class ClosestVote(Rebuilder): # ga naar closest met zelfde label, momenteel laten we alle vrijhoud van naar waar te bewegen
    """
    Dit idee is echt geinspireerd door de resultaten van LMNN-kNN (zie after)
    """
    def rebuild(self, repres, indices, data, represLabels): # represlabels enkel hier belangrijk
        k = 3
        if len(repres) < k:
            k = len(repres)
        model = KNeighborsClassifier(n_neighbors=k, weights='distance')
        model.fit(data[np.array(repres)], represLabels)

        predicted_label = model.predict(data[np.array(indices)])

        # Find the indices of the k nearest neighbors for the test sample
        _, n_indices = model.kneighbors(data[np.array(indices)])

        # Retrieve the labels of the neighbors
        neighbor_labels = np.array(represLabels)[n_indices]

        selection = neighbor_labels == predicted_label[..., None]
        # selection[:,1:] *=(np.diff(selection,axis=1)!=0)
        selection = selection.cumsum(axis=1).cumsum(axis=1) == 1 

        labels = n_indices[selection]

        for i in range(len(repres)): # repres wel nog altijd hun eigen nemen (dit is alleen nodig als twee repres overeenlappen)
            labels[np.array(indices) == repres[i]] = i # gaan er nu gewoon van uit dat het klopt

        return labels
    

class SemiCluster(Rebuilder): # TODO nog testen
    def rebuild(self, repres, indices, data, represLabels): # hier gaan we represselection voor nu negeren, REKENING MEEHOUDEN IN COBRAS
        # the inital centers
        centers = data[np.copy(repres)] # voor bestaande repres kan dit in theorie met de echte center (?)

        repers_indi = np.in1d(indices,repres)

        labels = np.zeros(len(indices))
        
        # alle repres initieel een gekend label
        for i in range(len(repres)):
            labels[indices == repres[i]] = i

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






        
        