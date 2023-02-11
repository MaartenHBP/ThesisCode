from abc import abstractmethod
import numpy as np
from sklearn.cluster import KMeans

class InstanceRebuilder: # hier kunnen de parent child relatie kapot gemaakt worden
    @abstractmethod
    def __init__(self, selection_strategy:str = 'all') -> None:
        self.selection_strategy = selection_strategy
    @abstractmethod
    def rebuildInstances(self, cobras, data, affinitymatrix): # returns new instances using the data of some original instances
        pass

    def SelectInstances(self, cobras, removeFromClusters = False): # if needed the superinstances are removed from there respective clusters
        '''
        Via hier worden de superinstances geselecteerd via de bijhorende strategie en returned ook van welke cluster die afkomstig zijn
        '''
        if self.selection_strategy == 'all':
            instances = cobras.clustering.get_superinstances()
            if removeFromClusters:
                cobras.clustering.clusters = [] # all clusters must be gone
            return instances # return all superinstances
        if self.selection_strategy == 'ClusterMostCL':
            pass # select the cluster with the most CL inside
        if self.selection_strategy == 'Wrong clusters':
            pass # select the clusters that have must-link to other clusters 
    def CreateNewInstances(self, cobras, removeFromClusters = False): # if removed from the cluster it needs to be put back in a new one
        pass # not really needed right now
################ Rebuild the existing superinstances ################################
class ClosestInstance(InstanceRebuilder): # deze nu al testen
    def __init__(self, selection_strategy: str = 'all') -> None:
        super().__init__(selection_strategy)

    def rebuildInstances(self, cobras, data, affinitymatrix):
        super = self.SelectInstances(cobras, False)
        repres = [s.get_representative_idx() for s in super]
        indices = []
        for s in super: indices.extend(s.indices) 
        closest_per_index = []

        for idx in indices:
            if idx in repres:
                closest_per_index.append(idx)
                continue
            closest = min(
                repres,
                key=lambda x: np.linalg.norm(
                    data[x] - data[idx]
                ),
            )
            closest_per_index.append(closest)
        closest_per_index = np.array(closest_per_index)
        indices = np.array(indices)
        for s in super:
            s.indices = (indices[closest_per_index == s.get_representative_idx()]).tolist()
            s.train_indices = [x for x in s.indices if x in cobras.train_indices]
            s.si_train_indices = [x for x in s.indices if x in cobras.train_indices]

        return False

# other algorithms that use the concept of "closest superinstances" -> calculate it using the points from the orinal superinstances transformed
################## Build new superinstances ##############################
class ClusterAgain(InstanceRebuilder):
    def __init__(self, selection_strategy: str = 'all', useCentres = False, k_strategy = 'instances') -> None:
        super().__init__(selection_strategy)
        self.useCentres = useCentres
        self.k_strategy = k_strategy

    def rebuildInstances(self, cobras, data, affinitymatrix):
        k = len(np.unique(cobras.clustering.construct_cluster_labeling())) if self.k_strategy == "clusters" else 0 # first cause next function deletes the clustering

        super = self.SelectInstances(cobras, True)

        if k == 0: k = len(super) # ff lelijk enzooo

        print(f"k = {k}")

        repres = np.array([s.get_representative_idx() for s in super]) if self.useCentres else None

    
        indices = np.arange(len(data))


        cluster = np.array(cobras.rebuild_cluster.cluster(data, indices, k, [], [], seed=cobras.random_generator.integers(1,1000000), affinity = affinitymatrix, centers = repres))
        
        
        new_supers = [cobras.create_superinstance(indices[cluster == i].tolist()) for i in set(cluster)]

        new_clusters = cobras.add_new_clusters_from_split(new_supers)

        cobras.clustering.clusters.extend(new_clusters)

        return True

class TopDownSplitting(InstanceRebuilder): # dit leek niet goed te werken
    def __init__(self, selection_strategy: str = 'all') -> None:
        super().__init__(selection_strategy)

    def rebuildInstances(self, cobras, data, affinitymatrix):
        # run COBRAS with this newly transformed data

        # first start with one big superinstance
        superinstances = [cobras.create_superinstance(
            list(range(data.shape[0]))
        )]

        # get the number of superinstances needed
        # nbInstances = len(cobras.clustering.get_superinstances())
        # print(nbInstances)
        nbInstances = 5

        for i in range(nbInstances):
            to_split = max(superinstances, key=lambda superinstance: len(superinstance.indices))
            new_super_instances = cobras.split_superinstance(to_split, 2, data = data)
            superinstances.remove(to_split)
            superinstances.extend(new_super_instances)

        new_clusters = cobras.add_new_clusters_from_split(superinstances)

        cobras.clustering.clusters = new_clusters

        return True

