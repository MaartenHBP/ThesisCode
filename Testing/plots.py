from abc import abstractmethod
import matplotlib.pyplot as plt
import pandas as pd

class Plotmaker:
    def __init__(self, name: str, path):
        self.name = name
        if path is not None:
            self.path = path
    
    # @abstractmethod
    # def addPoint(self):
    #     pass
    
    @abstractmethod
    def viewPlot(self):
        pass

    @abstractmethod
    def savePlot(self, path):
        pass

    @abstractmethod
    def savePoints(self):
        pass

    @abstractmethod
    def loadPoints(self):
        pass

class IRAPlot(Plotmaker):
    def __init__(self, name: str, colors = None, path = None):
        super().__init__(name, path)
        self.df = pd.DataFrame()

    def addPoint(self, series_name: str, points):
        self.df[series_name] = points

    def viewPlot(self, xLabel = "ARI"):
        self.df.plot(title=self.name, xlabel="Number of queries", ylabel=xLabel)

    