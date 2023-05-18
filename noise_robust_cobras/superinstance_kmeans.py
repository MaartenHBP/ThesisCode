import numpy as np

from noise_robust_cobras.superinstance import SuperInstance, SuperInstanceBuilder


class KMeans_SuperinstanceBuilder(SuperInstanceBuilder):
    def __init__(self):
        pass

    def makeSuperInstance(self, data, indices, train_indices, parent=None, centre = None):
        return SuperInstance_kmeans(data, indices, train_indices, parent, centre)


class SuperInstance_kmeans(SuperInstance): # hoe wordt de medoid bepaald
    def __init__(self, data, indices, train_indices, parent=None, centre = None):
        # super(SuperInstance_kmeans, self).__init__(data, indices, train_indices, parent)
        super(SuperInstance_kmeans, self).__init__(data, indices, train_indices, None)
        self.centroid = np.mean(data[indices, :], axis=0)
        self.si_train_indices = [x for x in indices if x in train_indices]
        self.parent = parent # jow wel nodig

        if len(set(self.si_train_indices)) < len(self.si_train_indices):
            print("something goes wrong!")
        try:
            # representative instance is the training instance that is closest to the clusters centroid
            if centre is None:
                self.representative_idx = min(
                    self.si_train_indices,
                    key=lambda x: np.linalg.norm(self.data[x, :] - self.centroid),
                )
            else:
                self.representative_idx = centre # TADA
        except:
            raise ValueError("Super instance without training instances")



    def distance_to(self, other_superinstance):
        return np.linalg.norm(self.centroid - other_superinstance.centroid)

    def copy(self):
        return SuperInstance_kmeans(self.data, self.indices, self.train_indices)

