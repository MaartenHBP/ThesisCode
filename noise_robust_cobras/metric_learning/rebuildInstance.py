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
            cobras.clustering.get_superinstances()
            if removeFromClusters:
                pass
            pass # return all superinstances
        if self.selection_strategy == 'ClusterMostCL':
            pass # select the cluster with the most CL inside
        if self.selection_strategy == 'Wrong clusters':
            pass # select the clusters that are 
    def CreateNewInstances(self, cobras, removeFromClusters = False): # if removed from the cluster it needs to be put back in a new one
        pass
################ Rebuild the existing superinstances ################################
class ClosestInstance(InstanceRebuilder): # deze nu al testen
    def __init__(self, selection_strategy: str = 'all') -> None:
        super().__init__(selection_strategy)

    def rebuildInstances(self, cobras, data, affinitymatrix):
        super = cobras.clustering.get_superinstances()
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

# other algorithms that use the concept of "closest superinstances" -> calculate it using the points from the orinal superinstances transformed
################## Build new superinstances ##############################
class ClusterAgain(InstanceRebuilder):
    def __init__(self, selection_strategy: str = 'all') -> None:
        super().__init__(selection_strategy)

    def rebuildInstances(self, cobras, data, affinitymatrix):
        super = cobras.clustering.get_superinstances()
        repres = np.array([s.get_representative_idx() for s in super])

        km = KMeans(len(super), init = data[repres], random_state=cobras.random_generator.integers(1,1000000))
        km.fit(data)
        indices = np.arange(len(data))

        cluster = np.array(km.labels_)
        
        new_supers = [cobras.create_superinstance(indices[cluster == i].tolist()) for i in set(cluster)]

        new_clusters = cobras.add_new_clusters_from_split(new_supers)

        cobras.clustering.clusters = new_clusters

        return True
