import numpy as np
from pathlib import Path
from abc import abstractmethod
from noise_robust_cobras.cobras import COBRAS
from noise_robust_cobras.querier.labelquerier import LabelQuerier
from noise_robust_cobras.metric_learning.metriclearning_algorithms import * 
from metric_learn import *
from pathlib import Path
import os
import pandas as pd
from sklearn.cluster import KMeans
import math
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors


dataset_path = Path('testing/datasets/created/a.data').absolute()
dataset = np.loadtxt(dataset_path, delimiter=',')
data = dataset[:, 1:]
target = dataset[:, 0]




AMOUNT_OF_LINKS_WANTED = 50
print("semisupervised constraints")
querier = LabelQuerier(None, target, AMOUNT_OF_LINKS_WANTED)
clusterer = COBRAS(correct_noise=False, seed=42)
all_clusters, runtimes, superinstances, clusterIteration, transformations, ml, cl = clusterer.fit(data, -1, None, querier)
# pairs = np.array(ml[0])
# print(pairs)
# constraints =  np.full(len(pairs), 1)
# pairs = np.vstack((ml,cl))
# constrains = np.full(len(ml) + len(cl), 1)
# constrains[len(ml):] = np.full(len(cl), -1)



# for i in range(20):

#     if i == 0:
#         continue

#     neigh = NearestNeighbors(n_neighbors=i)
#     neigh.fit(newdata)
#     distances, indices = neigh.kneighbors(newdata)


#     for ML in np.array(ml):


#         point1 = data[ML[0]]
#         point2 = data[ML[1]]

#         ind1 = np.array(indices[ML[0]])
#         ind2 = np.array(indices[ML[1]])



#         # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 1

#         newdata[ind1] = newdata[ind1] + 0.001*(np.ones((len(newdata[ind1]), 1)) * (point2 - point1)/norm(point1 - point2))
#         newdata[ind2] = newdata[ind2] + 0.001*(np.ones((len(newdata[ind2]), 1)) * (point1 - point2)/norm(point1 - point2))

#     for ML in np.array(cl):


#         point1 = data[ML[0]]
#         point2 = data[ML[1]]

#         ind1 = np.array(indices[ML[0]])
#         ind2 = np.array(indices[ML[1]])



#         # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 1

#         newdata[ind1] = newdata[ind1] + 0.1*(np.ones((len(newdata[ind1]), 1)) * (point1 - point2)/norm(point1 - point2))
#         newdata[ind2] = newdata[ind2] + 0.1*(np.ones((len(newdata[ind2]), 1)) * (point2 - point1)/norm(point1 - point2))


newdata = np.copy(data)
for i in range(500):
    

    for ML in np.array(ml):

        

        point1 = data[ML[0]]
        point2 = data[ML[1]]


        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 1
        angle_difference = np.nan_to_num(np.arccos(((data - point1) @ (point2 - point1)) / (norm(data - point1, axis = 1) * norm(point2 - point1))))

        indices = angle_difference < math.radians(5)

        indices[ML[0]] = False

        newdata[indices] = newdata[indices] + (0.0005 * np.reciprocal(1 + norm(data[indices] - point2, axis = 1)) * (math.radians(7) - angle_difference[indices])/math.radians(7))[:, np.newaxis] * (np.ones((len(newdata[indices]), 1)) * (point1 - point2)/norm(point1 - point2))



        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 2
        angle_difference2 = np.nan_to_num(np.arccos(((data - point2) @ (point1 - point2)) / (norm(data - point2, axis = 1) * norm(point1 - point2))))

        indices2 = angle_difference2 < math.radians(5)
        indices2[ML[1]] = False
        newdata[indices2] = newdata[indices2] + (0.0005 * np.reciprocal(1 + norm(data[indices2] - point1, axis = 1)) * (math.radians(7) - angle_difference2[indices2])/math.radians(7))[:, np.newaxis] * (np.ones((len(newdata[indices2]), 1)) * (point2 - point1)/norm(point1 - point2))

    for ML in np.array(cl):

        point1 = data[ML[0]]
        point2 = data[ML[1]]


        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 1
        angle_difference = np.nan_to_num(np.arccos(((data - point1) @ (point2 - point1)) / (norm(data - point1, axis = 1) * norm(point2 - point1))))

        indices = angle_difference < math.radians(7)

        indices[ML[0]] = False

        newdata[indices] = newdata[indices] + (0.001 * np.reciprocal(1 + norm(data[indices] - point2, axis = 1)) * (math.radians(7) - angle_difference[indices])/math.radians(7))[:, np.newaxis] * (np.ones((len(newdata[indices]), 1)) * (point2 - point1)/2)



        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 2
        angle_difference2 = np.nan_to_num(np.arccos(((data - point2) @ (point1 - point2)) / (norm(data - point2, axis = 1) * norm(point1 - point2))))

        indices2 = angle_difference2 < math.radians(7)
        indices2[ML[1]] = False
        newdata[indices2] = newdata[indices2] + (0.001 * np.reciprocal(1 + norm(data[indices2] - point1, axis = 1)) * (math.radians(7) - angle_difference2[indices2])/math.radians(7))[:, np.newaxis] * (np.ones((len(newdata[indices2]), 1)) * (point1 - point2)/2)  
    data = np.copy(newdata)

# alpha = np.zeros(len(data))
# # alpha[:] = 0.3

# alpha[indices] = 1
# alpha[indices2] = 1
fig, ax = plt.subplots()

colors = {-1: "r", 1: "g"}

ax.scatter(x = newdata[:, 0], y = newdata[:, 1], c = target)


# for i in range(len(pairs)):

# point1 = data[ML[0]]
# point2 = data[ML[1]]

# x_values = [point1[0], point2[0]]
# y_values = [point1[1], point2[1]]
# ax.plot(x_values, y_values, c = colors[1])

plt.show()



