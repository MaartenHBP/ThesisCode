from abc import abstractmethod
import numpy as np

class InstanceRebuilder:
    @abstractmethod
    def __init__(self) -> None:
        pass
    @abstractmethod
    def rebuildInstances(self, instances, data): # returns new instances using the data of some original instances
        pass