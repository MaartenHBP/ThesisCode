import copy
import gc
import itertools
import logging
from enum import Enum
from typing import Union

from metric_learn import NCA

import numpy as np

from noise_robust_cobras.cluster import Cluster
from noise_robust_cobras.clustering import Clustering
from noise_robust_cobras.clustering_algorithms.clustering_algorithms import (
    KMeansClusterAlgorithm,
    SemiKMeansClusterAlgorithm,
    KMedoidsCLusteringAlgorithm,
    ClusterAlgorithm,
)
from noise_robust_cobras.rebuild_algorithms.rebuild_algorithms import (
    SemiCluster,
    ClosestRebuild,
    ClosestVote,
    Rebuilder
)
from sklearn.cluster import KMeans
from noise_robust_cobras.cobras_logger import ClusteringLogger
from noise_robust_cobras.strategies.splitlevel_estimation import (
    StandardSplitLevelEstimationStrategy,
    ConstantSplitLevelEstimationStrategy
)
from noise_robust_cobras.strategies.superinstance_selection import (
    SuperinstanceSelectionHeuristic,
    MostInstancesSelectionHeuristic,
    LeastInstancesSelectionHeuristic,
)
from noise_robust_cobras.superinstance import SuperInstance, SuperInstanceBuilder
from noise_robust_cobras.superinstance_kmeans import KMeans_SuperinstanceBuilder
from noise_robust_cobras.noise_robust.datastructures.certainty_constraint_set import (
    NewCertaintyConstraintSet,
)
from noise_robust_cobras.noise_robust.datastructures.constraint import Constraint
from noise_robust_cobras.noise_robust.datastructures.constraint_index import (
    ConstraintIndex,
    ConstraintBlobs
)
from noise_robust_cobras.noise_robust.noise_robust_possible_worlds import (
    gather_extra_evidence,
)
from noise_robust_cobras.querier.querier import MaximumQueriesExceeded

from noise_robust_cobras.metric_learning.module.metriclearners import *

import random 

from sklearn.metrics.pairwise import euclidean_distances

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

from noise_robust_cobras.dynamicRadiuskNN import Radius_Nearest_Neighbors_Classifier 


class SplitResult(Enum):
    SUCCESS = 1
    NO_SPLIT_POSSIBLE = 2
    SPLIT_FAILED = 3


class COBRAS: # set seeds!!!!!!!!; als je clustert een seed setten door een random getal te genereren
    certainty_constraint_set: NewCertaintyConstraintSet
    clustering: Union[Clustering, None]

    def __init__(
        self,
        ###########################################################
        cluster_algo = "KMeansClusterAlgorithm",
        cluster_algo_parameters = {},
        # rebuild_cluster: ClusterAlgorithm = KMeansClusterAlgorithm,
        # rebuild_cluster_parameters = {},
        ###########################################################
        superinstance_builder: SuperInstanceBuilder = KMeans_SuperinstanceBuilder(),
        split_superinstance_selection_heur: SuperinstanceSelectionHeuristic = None,
        splitlevel_strategy=None,
        splitlevelInt = 4,
        noise_probability=0.10,
        minimum_approximation_order=2,
        maximum_approximation_order=3,
        certainty_threshold=0.95,
        seed=None, # seed is bekijken (nog niet gedaan)
        correct_noise=True,
        logger=None,
        cobras_logger=None,
        ###################
        # METRIC LEARNING #
        ###################
        metricLearner = "LMNN_wrapper",
        metricLearer_arguments = {},

        metricLevel = "all",
        metricSuperInstanceLevel = 0,
        changeToMedoids = False,

        ##########
        # During #
        ##########
        metricAmountQueriesAsked = 50,
        learnAMetric = False,
        metricInterval = 0,

        ###########
        # Initial #
        ###########
        initial = False,
        initialSupervised = 0, 
        initialRandom = True, 

        ###########
        # Rebuild #
        ########### 
        rebuildPhase = False,
        rebuildAmountQueriesAsked = 50,
        rebuildInterval = 0,

        rebuildLevel = "all", 
        rebuildSuperInstanceLevel = 0, # nul is enkel naar de superinstances apart kijken, vanaf nu gaan we van top-down naar beneden kijken

        rebuilder = 'ClosestRebuild',

        rebuildMetric = False,
        rebuilderKeepTransformed = False,

        #########
        # After #
        #########
        after = False,
        afterAmountQueriesAsked = 50,
        afterRadius = False,
        afterLambda = 1,
        

        after_k = 3,
        after_weights = "uniform",
        
        afterMetric = False, # standaard geen metriek leren
        afterKeepTransformed = False, # werken we met deze metriek verder

        afterLevel = "all", # kan nu ook de waarde combined hebben
        afterSuperInstanceLevel = 0,


        ####################
        # Constraint_index #
        ####################
        useNewConstraintIndex = False,
        mergeBlobs = False, # dit is voor wanneer er niet gemerged wordt op het niveau
        represBlobs = False, # enkel de repres
        plusBlobs = False


    ):

        self.seed = seed # seed is belangrijk!!!!!!!!!!!!!!!!!!!!!  

        # init data, querier, max_questions, train_indices and store_intermediate results
        # already initialised so object size does not change during execution
        # python can optimize
        self.data = None
        self.original = None
        self.querier = None
        self.train_indices = None

        # init cobras_cluster_algo
        self.cluster_algo: ClusterAlgorithm = eval(cluster_algo)(**cluster_algo_parameters)
        self.superinstance_builder = superinstance_builder
        # self.splitlevel_cluster = KMeansClusterAlgorithm(askExtraConstraints=False) # OBSOLETE, uiteindelijk niet meer aangekomen
        ###################
        # METRIC LEARNING #
        ###################
        self.metricLearner = eval(metricLearner) # wordt als string meegegeven 
        self.metricLearer_arguments = metricLearer_arguments
        self.metricLevel = metricLevel
        self.metricSuperInstanceLevel = metricSuperInstanceLevel
        self.changeToMedoids = changeToMedoids # OBSOLETE

        self.learedMetric = None

        ##########
        # During #
        ##########
        self.learnAMetric = learnAMetric
        self.metricAmountQueriesAsked = metricAmountQueriesAsked
        self.metricInterval = metricInterval # na hoeveel queries opnieuw een metriek leren


        ###########
        # Initial #
        ###########
        self.initial = initial
        self.initialSupervised = initialSupervised # is een percentage
        self.initialRandom = initialRandom

        ###########
        # Rebuild #
        ###########
        self.rebuildPhase = rebuildPhase
        self.rebuildAmountQueriesAsked = rebuildAmountQueriesAsked # vanaf hoeveel queries gevraagd voeren we dit uit 
        self.rebuildInterval = rebuildInterval
        
        self.rebuildLevel = rebuildLevel 
        self.rebuildSuperInstanceLevel = rebuildSuperInstanceLevel # nul is enkel naar de superinstances apart kijken, vanaf ! gaan we van top-down naar beneden kijken

        self.rebuilder = rebuilder

        self.rebuildMetric = rebuildMetric
        self.rebuilderKeepTransformed = rebuilderKeepTransformed
        

        #########
        # After #
        #########
        self.doAfter = after
        self.afterAmountQueriesAsked = 1

        self.after_k = after_k
        self.after_weights = after_weights

        self.afterMetric = afterMetric
        self.afterKeepTransformed = afterKeepTransformed

        self.afterLevel = afterLevel
        self.afterSuperInstanceLevel = afterSuperInstanceLevel
        self.afterRadius = afterRadius
        self.afterLambda = afterLambda

        ####################
        # Constraint_index #
        ####################
        self.useNewConstraintIndex = useNewConstraintIndex
        self.mergeBlobs = mergeBlobs
        self.represBlobs = represBlobs
        self.plusBlobs = plusBlobs

        # init split superinstance selection heuristic
        if split_superinstance_selection_heur is None:
            self.split_superinstance_selection_heur = MostInstancesSelectionHeuristic()
        else:
            self.split_superinstance_selection_heur = split_superinstance_selection_heur

        # init splitlevel_heuristic
        if splitlevel_strategy is None:
            self.splitlevel_strategy = StandardSplitLevelEstimationStrategy(
            LeastInstancesSelectionHeuristic()
        )
        else:
            if splitlevel_strategy == "constant":
                self.splitlevel_strategy = ConstantSplitLevelEstimationStrategy(splitlevelInt)
            else:
                self.splitlevel_strategy = splitlevel_strategy

        # variables used during execution
        self.clustering_to_store = None
        self.clustering = None
        self.random_generator = None

        # logging
        self._log = logging.getLogger(__name__) if logger is None else logger
        self._cobras_log = (
            ClusteringLogger() if cobras_logger is None else cobras_logger
        )

        # certainty_constraint_set
        if correct_noise:
            self.certainty_constraint_set: NewCertaintyConstraintSet = NewCertaintyConstraintSet(
                minimum_approximation_order,
                maximum_approximation_order,
                noise_probability,
                self._cobras_log,
            )
            self.constraint_index = self.certainty_constraint_set.constraint_index
        else:
            self.certainty_constraint_set = None
            self.constraint_index = ConstraintIndex()

        self.constraint_index_advanced = ConstraintBlobs() # new constraint index

        self.certainty_threshold = certainty_threshold

        self.correct_noise = correct_noise

    # LOGGER HERE!
    @property
    def clustering_logger(self):
        return self._cobras_log   

    def fit(self, X, nb_clusters, train_indices, querier):
        """
            Perform clustering.
            The number of clusters (nb_clusters) is not used in COBRAS but is added as a parameter to have a consistent
            interface over all clustering algorithms
            :param X: numpy array that where each row is an instance
            :param nb_clusters: IGNORED, COBRAS determines the amount of clusters dynamically
            :param train_indices: the indices for which COBRAS can ask constraints, if there is no training test_set use None
            :param querier: a Querier object that can answer queries about the data X
            :return: a tuple(all_clusters, runtimes, ml, cl) where all_clusters are the intermediate clusterings (for each query there is an intermediate clustering stored)
                runtimes is the time the algorithm has been executing after each query
                ml and cl are both lists of tuples representing the must-link and cannot-link constraints
                note: these are the constraints that we got from the user! So there might be noisy constraints in these lists!
        """
        self.random_generator = np.random.default_rng(self.seed) # hier random_generator gezet
        self._cobras_log.log_start_clustering()
        self.data = X

        self.original = np.copy(X)

        self.learedMetric = np.copy(self.data) # de huidige geleerde metriek
        self.metricCounter = 0


        self.train_indices = (
            train_indices if train_indices is not None else range(len(X)) # hier worden enkel queries van gevraagd
        )
        self.split_superinstance_selection_heur.set_clusterer(self) # nadenken over dit design
        self.splitlevel_strategy.set_clusterer(self)
        self.querier = querier

        # self.initialData = np.copy(X) -> not needed anymore

        # initial clustering: all instances in one superinstance in one cluster
        initial_superinstance = self.create_superinstance(
            list(range(self.data.shape[0]))
        )
        initial_clustering = Clustering([Cluster([initial_superinstance])])
        self.clustering = initial_clustering

        # ### SUPERINSTANCES ### -> deze logger nog fixen -> nog volgens de juiste manier maken
        # self._cobras_log.addSuperinstances(self.clustering.construct_superinstance_labeling())
        # self._cobras_log.addClus(np.copy(self.clustering.construct_cluster_labeling()))

        # last valid clustering keeps the last completely merged clustering
        last_valid_clustering = None


        ################
        # Metric learn #
        ################
        self.initial_transform()



        while not self.querier.query_limit_reached():
            
            # adapt metric phase
            self.learnMetricDuring()
            if self.querier.query_limit_reached():
                break

            # rebuild phase
            self.rebuild() 
            if self.querier.query_limit_reached():
                break

            # during this iteration store the current clustering + after
            a, b = self.after()
            if self.querier.query_limit_reached():
                break
            self._cobras_log.update_clustering_to_store(a, b) # das hier nog een artifact
            self.clustering_to_store = self.clustering.construct_cluster_labeling()

            
            # splitting phase
            self._cobras_log.log_entering_phase("splitting")
            statuscode = self.split_next_superinstance()
            if statuscode == SplitResult.NO_SPLIT_POSSIBLE:
                # there is no split left to be done
                # we have produced the best clustering
                break
            elif statuscode == SplitResult.SPLIT_FAILED:
                # tried to split a superinstance but failed to split it
                # this is recorded in the superinstance
                # we will split another superinstance in the next iteration
                continue

            # merging phase
            self._cobras_log.log_entering_phase("merging")

            if self.correct_noise:
                # make a copy of the current clustering and perform the merging phase on it
                clustering_copy = copy.deepcopy(self.clustering)
                fully_merged, new_user_constraints = self.merge_containing_clusters(
                    clustering_copy
                )
                corrected_clustering = None
                if fully_merged:
                    # if we fully merged we can confirm and correct the clustering
                    # if not the query limit is reached so we have to stop
                    try:
                        fully_merged, corrected_clustering = self.confirm_and_correct(
                            new_user_constraints, clustering_copy
                        )
                    except MaximumQueriesExceeded:
                        # if during the confirm and correct the query limit is reached fully_merged is false
                        fully_merged = False
                    self.clustering = corrected_clustering
                # explicit call to garbage collector to avoid memory problems
                gc.collect()
            else:
                fully_merged, _ = self.merge_containing_clusters(self.clustering)

            # correctly log intermediate results
            if fully_merged:
                a, b = self.after()
                if self.querier.query_limit_reached():
                    break
                self._cobras_log.update_last_intermediate_result(a, b) # a is op zich de belangrijkste, b dient voor als er plots moeten zijn

            # fill in the last_valid_clustering whenever appropriate
            # after initialisation or after that the current clustering is fully merged
            if fully_merged or last_valid_clustering is None:
                last_valid_clustering = copy.deepcopy(self.clustering)



        self.clustering = last_valid_clustering
        self._cobras_log.log_end_clustering()
        all_clusters = self._cobras_log.get_all_clusterings() 
        
        runtimes = self._cobras_log.get_runtimes()
        ml, cl = self._cobras_log.get_ml_cl_constraint_lists()


        return all_clusters, runtimes, ml, cl # volgorde van returnen is van belang

    ###########################
    #       SPLITTING         #
    ###########################

    def split_next_superinstance(self):
        """
            Execute the splitting phase:
                1) select the next super-instance to split
                2) split the super-instance into multiple smaller super-instances
        :return:
        """
        # identify the next superinstance to split
        to_split, originating_cluster = self.identify_superinstance_to_split()

        if to_split is None:
            return SplitResult.NO_SPLIT_POSSIBLE

        # remove to_split from the clustering
        # if self.keepSupervised:
        #     originating_cluster.super_instances.remove(to_split)
        #     originating_cluster.super_instances.append(self.create_superinstance([to_split.representative_idx], parent=to_split))
        #     to_split.indices.remove(to_split.representative_idx)
        #     to_split.train_indices.remove(to_split.representative_idx)
            
        # else:
        originating_cluster.super_instances.remove(to_split)
        if len(originating_cluster.super_instances) == 0:
            self.clustering.clusters.remove(originating_cluster)

        # split to_split into new clusters
        split_level = self.determine_split_level(to_split)
                
        new_super_instances = self.split_superinstance(to_split, split_level, determination=False)
        self._log.info(
            f"Splitted super-instance {to_split.representative_idx} in {split_level} new super-instances {list(si.representative_idx for si in new_super_instances)}"
        )

        new_clusters = self.add_new_clusters_from_split(new_super_instances)

        if not new_clusters:
            # it is possible that splitting a super-instance does not lead to a new cluster:
            # e.g. a super-instance constains 2 points, of which one is in the test set
            # in this case, the super-instance can be split into two new ones, but these will be joined
            # again immediately, as we cannot have super-instances containing only test points (these cannot be
            # queried)
            # this case handles this, we simply add the super-instance back to its originating cluster,
            # and set the already_tried flag to make sure we do not keep trying to split this superinstance
            self._log.info("Split failed! restoring original state")
            originating_cluster.super_instances.append(to_split)
            to_split.tried_splitting = True
            to_split.children = None

            if originating_cluster not in self.clustering.clusters:
                self.clustering.clusters.append(originating_cluster)

            return SplitResult.SPLIT_FAILED
        else:
            self.clustering.clusters.extend(new_clusters)

        return SplitResult.SUCCESS

    def identify_superinstance_to_split(self):
        """
            Identify the next super-instance that needs to be split using the split superinstance selection heuristic
            :return: (the super instance to split, the cluster from which the super instance originates)
        """
        # if there is only one superinstance return that superinstance as superinstance to split
        if (
            len(self.clustering.clusters) == 1
            and len(self.clustering.clusters[0].super_instances) == 1
        ):
            return (
                self.clustering.clusters[0].super_instances[0],
                self.clustering.clusters[0],
            )

        options = []
        for cluster in self.clustering.clusters:

            if cluster.is_pure:
                continue

            if cluster.is_finished:
                continue

            for superinstance in cluster.super_instances:
                if superinstance.tried_splitting:
                    continue

                if len(superinstance.indices) == 1:
                    continue

                if len(superinstance.train_indices) < 2:
                    continue

                else:
                    options.append(superinstance)

        if len(options) == 0:
            return None, None

        superinstance_to_split = self.split_superinstance_selection_heur.choose_superinstance(
            options
        )
        originating_cluster = [
            cluster
            for cluster in self.clustering.clusters
            if superinstance_to_split in cluster.super_instances
        ][0]

        if superinstance_to_split is None:
            return None, None

        return superinstance_to_split, originating_cluster

    def determine_split_level(self, superinstance):
        """
            Determine the splitting level to split the given super-instance
        """
        return self.splitlevel_strategy.estimate_splitting_level(superinstance)

    def split_superinstance(self, si, k, determination = True): # data is for rebuildclustering to call it (askMetric kan nu in theorie ook weg)
        """
            Actually split the given super-instance si in k (the splitlevel) new super-instances

            note: if splitting with self.cluster_algo results in a super-instance that has no training_instances,
            this super-instance is merged with another super-instance that does still have training instances
            :param si: the super-instance to be split
            :param k: the splitlevel to be used
            :return:   A list with the resulting super-instances
            :rtype List[Superinstance]
        """

        # cluster the instances of the superinstance
        clusters, medoids = self.cluster_algo.cluster( # ML and CL meegeven
            np.copy(self.data), si.indices, k, [], [], seed=self.random_generator.integers(1,1000000), cobras = self,
            transformed=self.original # seed voor clusteren wordt rangom gegenereerd (zo krijgen we altijd dezelfde resultaten) => zo moet seed gedaan worden (ook als we ITML enzo oproepen)

        )

        # based on the resulting clusters make new superinstances
        # superinstances with no training instances are assigned to the closest superinstance with training instances -> euhhh
        training = []
        no_training = []
        for new_si_idx in set(clusters):
            cur_indices = [
                si.indices[idx] for idx, c in enumerate(clusters) if c == new_si_idx
            ]

            si_train_indices = [x for x in cur_indices if x in self.train_indices]
            if len(si_train_indices) != 0:
                training.append(self.create_superinstance(cur_indices, si, medoids[new_si_idx] if medoids is not None else None)) # hier wordt de parent gezet
            else:
                no_training.append(
                    (cur_indices, np.mean(self.data[cur_indices, :], axis=0))
                )

        for indices, centroid in no_training:
            closest_train = min(
                training,
                key=lambda x: np.linalg.norm(
                    self.data[x.representative_idx, :] - centroid
                ),
            )
            closest_train.indices.extend(indices)

        si.children = training 
        return training

    @staticmethod
    def add_new_clusters_from_split(si):
        """
            small helper function: adds the new super-instances to the current clustering each in their own cluster

        """
        new_clusters = []
        for x in si:
            new_clusters.append(Cluster([x]))

        if len(new_clusters) == 1:
            return None
        else:
            return new_clusters

    ###########################
    #        MERGING          #
    ###########################

    def merge_containing_clusters(self, clustering_to_merge):
        """
            Perform the merging step to merge the clustering together

        :param clustering_to_merge:
        :return:
        """
        query_limit_reached = False
        merged = True

        # the set of new user constraints that are used during merging
        new_user_constraints = set()

        while merged and not self.querier.query_limit_reached():

            clusters_to_consider = [
                cluster
                for cluster in clustering_to_merge.clusters
                if not cluster.is_finished
            ]

            cluster_pairs = itertools.combinations(clusters_to_consider, 2)
            cluster_pairs = [
                x
                for x in cluster_pairs
                if not self.cannot_link_between_clusters(
                    x[0], x[1], new_user_constraints
                )
            ]
            cluster_pairs = sorted(cluster_pairs, key=lambda x: x[0].distance_to(x[1]))

            merged = False
            for x, y in cluster_pairs:

                if self.querier.query_limit_reached():
                    query_limit_reached = True
                    break

                # we will reuse or get a new constraint
                constraint = self.get_constraint_between_clusters(x, y, "merging")
                new_user_constraints.add(constraint)
                if constraint.is_ML():
                    x.super_instances.extend(y.super_instances)
                    clustering_to_merge.clusters.remove(y)
                    merged = True
                    break

        fully_merged = not query_limit_reached and not merged

        return fully_merged, new_user_constraints

    def cannot_link_between_clusters(self, c1, c2, new_constraints):
        # first check if we can reuse from the constraint_structure itself
        reused = self.check_constraint_reuse_clusters(c1, c2)
        if reused is not None:
            if reused.is_CL():
                new_constraints.add(reused)
                return True
            return False

        # otherwise check if we can reuse from new_constraints
        for s1, s2 in itertools.product(c1.super_instances, c2.super_instances):
            if (
                Constraint(s1.representative_idx, s2.representative_idx, False)
                in new_constraints
            ):
                return True
        return False

    def must_link_between_clusters(self, c1, c2, new_constraints):
        # first check if we can reuse from the constraint_structure itself
        reused = self.check_constraint_reuse_clusters(c1, c2)
        if reused is not None:
            return reused.is_ML()

        # otherwise check if we can reuse fron new_constraints
        for s1, s2 in itertools.product(c1.super_instances, c2.super_instances):
            if (
                Constraint(s1.representative_idx, s2.representative_idx, True)
                in new_constraints
            ):
                return True
        return False

    ######################################################
    ########### handling noisy constraints ###############
    ######################################################

    def confirm_and_correct(self, new_user_constraints, clustering_copy):
        """
            Confirm and correct the relevant user constraints
            :param new_user_constraints:
            :param clustering_copy:
            :return:
        """

        fully_merged = True

        while len(new_user_constraints) > 0 and fully_merged:
            # gather extra evidence for the uncertain userconstraints used during merging
            relevant_instances = self.clustering.get_si_representatives()
            all_relevant_constraints = self.constraint_index.find_constraints_between_instance_set(
                relevant_instances
            )
            noisy_detected = gather_extra_evidence(
                self.certainty_constraint_set,
                all_relevant_constraints,
                self.certainty_threshold,
                self.querier,
                self._cobras_log,
            )

            # if no noise detected continue with the next iteration of COBRAS
            if not noisy_detected:
                break

            # there is noise but this could also be noise in userconstraints from previous iterations!
            # so start from a clustering where each super-instance is in its own cluster!
            self._cobras_log.log_entering_phase("merging")
            all_sis = self.clustering.get_superinstances()
            clusters = [Cluster([si]) for si in all_sis]
            clustering_copy = Clustering(clusters)
            fully_merged, new_user_constraints = self.merge_containing_clusters(
                clustering_copy
            )

        if fully_merged:
            # log constraints used during clustering
            relevant_instances = self.clustering.get_si_representatives()
            all_relevant_constraints = self.constraint_index.find_constraints_between_instance_set(
                relevant_instances
            )
            self._cobras_log.log_corrected_constraint_set(all_relevant_constraints)

        return fully_merged, clustering_copy

    ########
    # util #
    ########
    def get_constraint_length(self):
        return self.constraint_index.get_number_of_constraints()

    def create_superinstance(self, indices, parent=None, medoid = None) -> SuperInstance:
        return self.superinstance_builder.makeSuperInstance(
            self.data, indices, self.train_indices, parent, medoid
        )

    ############################################
    # constraint querying and constraint reuse #
    ############################################

    def get_constraint_between_clusters(self, c1: Cluster, c2: Cluster, purpose):
        """
            Gets a constraint between clusters c1 and c2
            If there is already a known constraint between these two clusters it is reused
            otherwise a new constraint between the 2 clusters is queried
        :param c1: the first cluster
        :param c2: the second cluster
        :param purpose: the purpose of this constraint
        :return: the reused or new constraint
        """
        reused_constraint = self.check_constraint_reuse_clusters(c1, c2)
        if reused_constraint is not None:
            return reused_constraint
        si1, si2 = c1.get_comparison_points(c2)
        return self.query_querier(
            si1.representative_idx, si2.representative_idx, purpose
        )

    def get_constraint_between_superinstances(
        self, s1: SuperInstance, s2: SuperInstance, purpose
    ):
        """
            Gets a constraint between the representatives of superinstances s1 and s2
            If there is already a known constraint this constraint is reused
            otherwise a new constraint between the super-instance representatives is queried
        :param s1: the first super-instance
        :param s2: the second super-instance
        :param purpose: the purpose of this constraint
        :return: the reused or new constraint
        """
        reused_constraint = self.check_constraint_reuse_between_representatives(s1, s2)
        if reused_constraint is not None:
            return reused_constraint
        return self.query_querier(s1.representative_idx, s2.representative_idx, purpose)

    def get_constraint_between_instances(self, instance1, instance2, purpose):
        """
            Gets a constraint between the instances instance1 and instance 2
            If there is already a known constraint between these instances that constraint is reused
            otherwise a new constraint between the instances is queried
        :param instance1: the first instance
        :param instance2: the second instance
        :param purpose: the purpose of this constraint
        :return: the reused or new constraint
        """
        reused_constraint = self.check_constraint_reuse_between_instances(
            instance1, instance2
        )

        if reused_constraint is not None:
            return reused_constraint

        min_instance = min(instance1, instance2)
        max_instance = max(instance1, instance2)
        return self.query_querier(min_instance, max_instance, purpose)

    def check_constraint_reuse_clusters(self, c1: Cluster, c2: Cluster):
        """
            Checks whether or not there is a known constraint between clusters c1 and c2
            if there is return this constraint otherwise return None
        :param c1: the first cluster
        :param c2: the second cluster
        :return: the existing constraint if there is one, none otherwise
        :rtype Union[Constraint, None]
        """
        superinstances1 = c1.super_instances
        superinstances2 = c2.super_instances

        for si1, si2 in itertools.product(superinstances1, superinstances2):
            reused_constraint = self.check_constraint_reuse_between_representatives(
                si1, si2
            )
            if reused_constraint is not None:
                return reused_constraint

        return None

    def check_constraint_reuse_superinstances(self, si1, si2):
        """
            Checks whether or not there is a known constraint between the representatives of si1 and si2
            if there is return this constraint otherwise return None
        :param si1: the first super-instance
        :param si2: the second super-instance
        :return: the existing constraint if there is one, none otherwise
        :rtype Union[Constraint, None]
        """
        reused_constraint = self.check_constraint_reuse_between_representatives(
            si1, si2
        )
        return reused_constraint

    def check_constraint_reuse_between_representatives(self, si1, si2):
        """
               Checks whether or not there is a known constraint between the representatives of si1 and si2
               if there is return this constraint otherwise return None
           :param si1: the first super-instance
           :param si2: the second super-instance
           :return: the existing constraint if there is one, none otherwise
           :rtype Union[Constraint, None]
        """
        return self.check_constraint_reuse_between_instances(
            si1.representative_idx, si2.representative_idx
        )

    def check_constraint_reuse_between_instances(self, i1, i2):
        """
            Checks whether or not there is a known constraint between the instances i1 and i2
            if there is return this constraint otherwise return NOne
        :param i1: the first instance
        :param i2: the second instance
        :return: the existing constraint if there is one, none otherwise
        :rtype Union[Constraint, None]
        """

        if self.useNewConstraintIndex:
            # clust = np.array(self.clustering_to_store)
            # options = np.array(list(range(self.data.shape[0])))
            # if len(clust) > 0:
                # return self.constraint_index_advanced.checkReuse(i1, i2, self.data, options[clust == clust[i1]], options[clust == clust[i2]], self.plusBlobs)
            return self.constraint_index_advanced.checkReuse(i1, i2)
        else:
            reused_constraint = None
            ml_constraint = Constraint(i1, i2, True)
            cl_constraint = Constraint(i1, i2, False)
            constraint_index = self.constraint_index

            if ml_constraint in constraint_index:
                reused_constraint = ml_constraint
            elif cl_constraint in constraint_index:
                reused_constraint = cl_constraint

            # if reused_constraint is not None:
            #     self._cobras_log.log_reused_constraint_instances(reused_constraint.is_ML(), i1, i2)
            return reused_constraint
    
    def simple_query_querier(self, instance1, instance2): # Dit is voor de merging phase in constraint_index, moet enkel een con straint meegeven
        if self.querier.query_limit_reached():
            print("going over query limit! ", self.get_constraint_length())
        # print("query ",self.get_constraint_length())
        min_instance = min(instance1, instance2)
        max_instance = max(instance1, instance2)
        constraint_type = self.querier._query_points(min_instance, max_instance)

        self._cobras_log.log_new_user_query(
                Constraint(min_instance, max_instance, constraint_type, purpose="simple")
            )

        return Constraint(min_instance, max_instance, constraint_type)


    def query_querier(self, instance1, instance2, purpose):
        """
            Function to query the querier
            The constraint obtained from the querier is stored in
            the certainty_constraint set or constraint_index (depending on whether correct noise is true or false)

            This method should not be called if check_constraint_reuse_between_instances(i1,i2) returns a constraint

        :param instance1: the first instance
        :param instance2: the second instance
        :param purpose: the purpose of this query
        :return:
        """
        if self.querier.query_limit_reached():
            print("going over query limit! ", self.get_constraint_length())
        # print("query ",self.get_constraint_length())
        min_instance = min(instance1, instance2)
        max_instance = max(instance1, instance2)
        constraint_type = self.querier._query_points(min_instance, max_instance)


        if self.useNewConstraintIndex: # new advanced yeet
            new_constraint = Constraint(min_instance, max_instance, constraint_type, purpose=purpose)
            self.constraint_index_advanced.addConstraints(new_constraint)

            self._cobras_log.log_new_user_query(
                Constraint(min_instance, max_instance, constraint_type, purpose=purpose) # voor de veiligheid een nieuwe aanmaken
            )
            return new_constraint

        else:
            if self.correct_noise: # IS NIET BELANRGIJK VOOR DE THESIS
                # add the new constraint to the certainty constraint set
                self.certainty_constraint_set.add_constraint(
                    Constraint(min_instance, max_instance, constraint_type, purpose=purpose)
                )
                new_constraint = next(
                    self.certainty_constraint_set.constraint_index.find_constraints_between_instances(
                        min_instance, max_instance
                    ).__iter__()
                )
            else:
                self.constraint_index.add_constraint(
                    Constraint(min_instance, max_instance, constraint_type, purpose=purpose)
                )
                new_constraint = next(
                    self.constraint_index.find_constraints_between_instances(
                        min_instance, max_instance
                    ).__iter__()
                )

            self._cobras_log.log_new_user_query(
                Constraint(min_instance, max_instance, constraint_type, purpose=purpose)
            )

            return new_constraint

    ##################
    # Learn a metric # 
    ##################

    def learnMetric(self):
        # print(self.metricCounter)
        # print(len(self._cobras_log.all_user_constraints))
        # print("==========")  
        levels = self.getFinegrainedLevel(self.metricLevel, self.metricSuperInstanceLevel)
        data = np.copy(self.data)

        labelled, clust, finished = self.constraint_index_advanced.cluster(self)

        if(len(labelled) == self.metricCounter): # als het dezelfde labels krijgt, ja dan moet je geen metriek leren => gaan er vanuit dat het aantal gelabelde instances stijgt
            return self.learedMetric
        
        self.metricCounter = len(labelled)

        if finished: # tis gedaan, het boeit ni meer
            return data

        for level in levels:
            indices = []
            repres = []
            for superinstance in level:
                for idx in superinstance.indices:
                    indices.append(idx)
                    if idx in labelled:
                        repres.append(idx)
            data[np.array(indices)] = self.metricLearner(preprocessor = np.copy(self.data), **self.metricLearer_arguments, seed = self.seed).fit_transform(None, None, np.copy(repres), clust[np.array(repres)])[np.array(indices)] # ff een poging ondernemen
        self.learedMetric = np.copy(data)
        
        return self.learedMetric



    ############################################
    # get superinstances per finegrained level #
    ############################################
    def getFinegrainedLevel(self,level:str, superinstanceLevel):
        if level == "all":
            return [self.clustering.get_superinstances()]
        if level == "cluster": 
            return self.clustering.get_superinstances_per_cluster()
        
        if level == "superinstance": # level up, wordt de standaard
            if superinstanceLevel == 0:
                return [[x] for x in self.clustering.get_superinstances()]
            
            supers = [self.clustering.get_superinstances()[0].get_root()]
            depth = supers[0].getEqualDepth() - superinstanceLevel
            if depth > 1:
                for i in range(depth - 1):
                    new = []
                    for s in supers:
                        new.extend(s.get_childeren())
                    supers = new
            
            superchilds = []
            for j in supers:
                superchilds.append(j.get_leaves())
            return superchilds
            

                
        else:
            return [self.clustering.get_superinstances()] # is equivalent aan all

    ###########
    # Initial #
    ###########
    def initial_transform(self):
        if not self.initial:
            return
        pairs = None # is hier nog voor legacy reasons
        y = None
        points = None
        labels = None

        if self.initialRandom:  
            points, labels =  self.querier.getRandomLabels(math.ceil(self.initialSupervised*len(self.data)), self.seed)
        else: # mss dit niet nodig
            querier = self.querier.createCopy(self.initialSupervised)
            clusterer = COBRAS(correct_noise=False, seed=self.seed)
            clusterer.fit(np.copy(self.data), -1, None, querier)

            points, clust, finished = self.constraint_index_advanced.cluster(self) # TODO
            labels = clust[points]
            
        self.data = self.metricLearner(preprocessor = np.copy(self.data), **self.metricLearer_arguments, seed = self.seed).fit_transform(np.copy(pairs), np.copy(y), np.copy(points), np.copy(labels))
        
    ##########
    # During # 
    ########## 
    def learnMetricDuring(self):
        if self.learnAMetric and len(self._cobras_log.all_user_constraints) >= self.metricAmountQueriesAsked:
            self.data = self.learnMetric()
            if self.metricInterval > 0:
                self.metricAmountQueriesAsked = len(self._cobras_log.all_user_constraints) + self.metricInterval # voor de volgende keer dat het mag uitgevoerd worden
            else:
                self.learnAMetric = False # er moet geen metriek meer geleerd worden
            if self.changeToMedoids:
                self.cluster_algo = SemiKMeansClusterAlgorithm()

    ##############
    # Rebuilding #
    ##############
       

    def rebuild(self):
        if not self.rebuildPhase or len(self._cobras_log.all_user_constraints) < self.rebuildAmountQueriesAsked:
            return
        levels = self.getFinegrainedLevel(self.rebuildLevel, self.rebuildSuperInstanceLevel)

        labelled, clust, finished = self.constraint_index_advanced.cluster(self)

        if finished:
            return

        clusters = [Cluster([]) for i in np.unique(clust)] # de nieuwe clusters, is evenveel als dat er labels zijn

        data = self.learnMetric() if self.rebuildMetric else np.copy(self.data)

        rebuilder = eval(self.rebuilder)()
            
        
        if self.rebuilderKeepTransformed: # werk later verder met deze transformatie
            self.data = np.copy(data)

        for level in levels: 

            indices = []
            repres = []

            for superinstance in level:
                for idx in superinstance.indices:
                    indices.append(idx)
                    if idx in labelled:
                        repres.append(idx)

            labels_repres = rebuilder.rebuild(repres, indices, data, represLabels=np.array(clust)[np.array(repres)]) 

            # bouw de superinstances en voeg aan de juiste cluster toe
            for i in range(len(indices)):
                idx = indices[i]
                if idx in repres: # dit punt moet een representatieve worden
                    points = np.array(indices)[labels_repres == labels_repres[i]] # zelfde label als een mogelijke representatieve volgen
                    superinstance = self.superinstance_builder.makeSuperInstance(
                        self.data, points.tolist(), self.train_indices, None
                    )
                    superinstance.representative_idx = idx # manueeel repres zetten
                    clusters[clust[idx]].super_instances.append(superinstance) # voeg aan de juiste cluster toe
            

        if self.rebuildInterval > 0:
            self.rebuildAmountQueriesAsked = len(self._cobras_log.all_user_constraints) + self.rebuildInterval
        else:
            self.rebuildPhase = False # moet maar een keer uitgvoerd worden
        self.clustering.clusters = clusters


    #########
    # AFTER #
    #########
    def after(self):

        labelled, clust, finished = self.constraint_index_advanced.cluster(self)
        
        if not self.doAfter or len(self._cobras_log.all_user_constraints) < self.afterAmountQueriesAsked or finished:
            return clust, self.clustering.get_superinstances() # geef gewoon terug wat er was
        
        levels = self.getFinegrainedLevel(self.afterLevel, self.afterSuperInstanceLevel)
        
        data = self.learnMetric() if self.afterMetric else self.data

        if self.afterKeepTransformed:
            self.data = np.copy(data)
        
        new = np.zeros(len(self.data)) # de nieuwe labels

        for level in levels:

            indices = []
            repres = []
            distances = []
            for superinstance in level:
                for idx in superinstance.indices:
                    indices.append(idx)
                    if idx in labelled:
                        repres.append(idx)
                        distances.append(1)
                    else:
                        distances.append(np.linalg.norm(data[idx] - data[superinstance.representative_idx]) * self.afterLambda)

                
            if self.afterRadius:
                n_neighbors = min(len(repres), self.after_k)
                model = Radius_Nearest_Neighbors_Classifier(r = distances, weights=self.after_weights) # een beetje een hacky maniers
                model.fit(np.array(data)[np.array(repres)], clust[np.array(repres)])
                new[np.array(indices)] = model.predict(np.array(data)[np.array(indices)]) # predict de labels
            else:
                n_neighbors = min(len(repres), self.after_k)
                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=self.after_weights)
                model.fit(np.array(data)[np.array(repres)], clust[np.array(repres)])
                new[np.array(indices)] = model.predict(np.array(data)[np.array(indices)]) # predict de labels


        new[np.array(labelled)] = clust[np.array(labelled)] # die zijn geweten en moeten zo blijven
        return new, self.clustering.get_superinstances() # return de nieuze labeling
    