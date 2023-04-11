import collections


class Clustering:
    def __init__(self, clusters):
        self.clusters = clusters

    def reset(self): # for rebuilding the superinstances (not needed wanneer de representatives niet veranderen)
        self.clusters = []

    def construct_cluster_labeling(self):

        pts_per_cluster = [cluster.get_all_points() for cluster in self.clusters]

        pred = [-1] * sum([len(x) for x in pts_per_cluster])

        for i, pts in enumerate(pts_per_cluster):
            for pt in pts:
                pred[pt] = i

        return pred

    def construct_superinstance_labeling(self):

        pts_per_cluster = []

        for cluster in self.clusters:
            pts_per_cluster.extend(cluster.get_all_points_per_superinstance())

        pred = [-1] * sum([len(x) for x in pts_per_cluster])

        for i, pts in enumerate(pts_per_cluster):
            for pt in pts:
                pred[pt] = i

        return pred

    def get_si_representatives(self):
        """
            Returns the indices of all the instances that are a super-instance representative in the current clustering
            :rtype: List[int]
        """
        result = set()
        for cluster in self.clusters:
            result.update(si.representative_idx for si in cluster.super_instances)
        return result

    def get_superinstances(self):
        result = []
        for cluster in self.clusters:
            result.extend(cluster.super_instances)
        return result
    
    def get_superinstances_per_cluster(self):
        result = []
        for cluster in self.clusters:
            result.append(cluster.super_instances)
        return result

    def create_generalized_super_instances(self, si):
        """
        Collects a list of 'generalized super-instances'.
        COBRAS always splits a super-instances in at least two new super-instances.
        If there is a must-link between these super-instances, and similarly a must-link between all the future splits
        of these super-intances, there is no need to consider the points as belonging to conceptually different super-
        instances (i.e. super-instances corresponding to different behaviour).
        This procedure constructs generalized super-instances: super-instances (i.e. leaves) that are part of a
        subtree with only must-links amongst eachoter are collected into a list.
        :param si: the root super-instance
        :return: a list of lists of super-instances, each entry in the list corresponds to one generalized super-instance,
        that may contain several super-instances
        """
        leaves = si.get_leaves()

        all_in_same_cluster = True
        cur_cluster = None
        for c in self.clusters:
            if leaves[0] in c.super_instances:
                cur_cluster = c
                break

        for l in leaves:
            if l not in cur_cluster.super_instances:
                all_in_same_cluster = False
                break

        if all_in_same_cluster:
            return [leaves]
        else:
            generalized_leaves = []
            for l in si.children:
                generalized_leaves.extend(self.create_generalized_super_instances(l))
            return generalized_leaves

    def get_cluster_to_generalized_super_instance_map(self):
        # first get the generalized super-instances
        generalized_super_instance_sets = self.create_generalized_super_instances(
            self.clusters[0].super_instances[0].get_root()
        )

        # now map each cluster to its leaves
        cluster_to_si = collections.defaultdict(list)
        for cluster in self.clusters:
            to_delete = []
            for l in generalized_super_instance_sets:
                if l[0] in cluster.super_instances:
                    cluster_to_si[cluster].append(l)
                    to_delete.append(l)

            for l in to_delete:
                generalized_super_instance_sets.remove(l)

        all_instances_ct = 0
        for k in cluster_to_si:
            for l in cluster_to_si[k]:
                for x in l:
                    all_instances_ct += len(x.indices)

        return cluster_to_si
