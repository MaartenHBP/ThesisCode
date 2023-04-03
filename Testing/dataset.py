
import warnings
warnings.filterwarnings("ignore") 
from IPython.core.display import display, HTML
import time
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from pathlib import Path
from sklearn.datasets import make_blobs
# random_state = 21
# X_mn, y_mn = make_moons(1000, noise=.1, random_state=random_state)

# fig, ax = plt.subplots(figsize=(9,7))
# ax.set_title('Data with ground truth labels - linear separation not possible', fontsize=18, fontweight='demi')
# ax.scatter(X_mn[:, 0], X_mn[:, 1],c=y_mn)

# df  = np.zeros(3*1000).reshape(1000, 3)
# df[:,0] = y_mn
# df[:,1:] = X_mn


# resulting_path = Path("datasets/created/a.data").absolute()
# np.savetxt(resulting_path, df, delimiter=',')

# from sklearn.cluster import SpectralClustering
# import numpy as np
# import scipy.sparse as sp
# X = np.array([[1, 1], [2, 1], [1, 0],
#                [4, 7], [3, 5], [3, 6]])
# clustering = SpectralClustering(n_clusters=2, eigen_solver="arpack", affinity='nearest_neighbors', n_neighbors=2).fit(X)

# aff = clustering.affinity_matrix_
# aff[0,0] = 0


# print('yeet')

x, _ = make_blobs(n_samples=400, centers=1, cluster_std=1.5)

print("yeet")

y = np.zeros(400)

m = np.mean(x, axis=0)

y[x[:,0] > m[0]] = 1

print(y)


df  = np.zeros(3*400).reshape(400, 3)
df[:,0] = y
df[:,1:] = x


resulting_path = Path("testing/datasets/created/oneblob.data").absolute()
np.savetxt(resulting_path, df, delimiter=',')
