from abc import abstractmethod
import matplotlib.pyplot as plt

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
        self.series = dict()

    def addPoint(self, series_name: str, x, y):
        samples = self.series.get(series_name, None)
        if samples is None:
            self.series[series_name] = ([],[])
        self.series[series_name][0].append(x)
        self.series[series_name][1].append(y)

    def viewPlot(self, xLabel = "ARI"):
        for name,samples in self.series.items():
            plt.plot(samples[0], samples[1], label=name)
        plt.legend(loc='lower right')
        plt.title(self.name)
        plt.xlabel("Number of queries")
        plt.ylabel(xLabel)
        plt.show()

    