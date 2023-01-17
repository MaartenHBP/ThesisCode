from abc import abstractmethod
import numpy as np
from noise_robust_cobras.cobras import *

class InstanceRebuilder: # hier kunnen de parent child relatie kapot gemaakt worden
    @abstractmethod
    def __init__(self, selection_strategy:str = 'all') -> None:
        self.selection_strategy = selection_strategy
    @abstractmethod
    def rebuildInstances(self, cobras, data, affinitymatrix): # returns new instances using the data of some original instances
        pass

    def SelectInstances(self, cobras:COBRAS, removeFromClusters = False): # if needed the superinstances are removed from there respective clusters
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
    def CreateNewInstances(self, cobras:COBRAS, removeFromClusters = False): # if removed from the cluster it needs to be put back in a new one
        pass
################ Rebuild the existing superinstances ################################
class ClosestInstance(InstanceRebuilder):
    def __init__(self, selection_strategy: str = 'all') -> None:
        super().__init__(selection_strategy)

    def rebuildInstances(self, cobras:COBRAS, data, affinitymatrix):
        return super().rebuildInstances(cobras, data, affinitymatrix)
# other algorithms that use the concept of "closest superinstances" -> calculate it using the points from the orinal superinstances transformed
################## Build new superinstances ##############################
class ClusterAgain(InstanceRebuilder):
    def __init__(self, selection_strategy: str = 'all') -> None:
        super().__init__(selection_strategy)

    def rebuildInstances(self, cobras:COBRAS, data, affinitymatrix):
        return True # cause the merge phase need to happen again
