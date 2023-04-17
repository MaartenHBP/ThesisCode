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
    ClusterAlgorithm,
)
from noise_robust_cobras.rebuild_algorithms.rebuild_algorithms import (
    SemiCluster,
    ClosestRebuild,
    Rebuilder
)
from sklearn.cluster import KMeans
from noise_robust_cobras.cobras_logger import ClusteringLogger
from noise_robust_cobras.strategies.splitlevel_estimation import (
    StandardSplitLevelEstimationStrategy,
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
)
from noise_robust_cobras.noise_robust.noise_robust_possible_worlds import (
    gather_extra_evidence,
)
from noise_robust_cobras.querier.querier import MaximumQueriesExceeded

from noise_robust_cobras.metric_learning.module.metriclearners import *

import random 

from sklearn.neighbors import KNeighborsClassifier


class SplitResult(Enum):
    SUCCESS = 1
    NO_SPLIT_POSSIBLE = 2
    SPLIT_FAILED = 3


class COBRAS: # set seeds!!!!!!!!; als je clustert een seed setten door een random getal te genereren
    certainty_constraint_set: NewCertaintyConstraintSet
    clustering: Union[Clustering, None] # aha met die union (kan ook nuttig zijn voor bepaalde string opties TODO

    def __init__(
        self,
        ###########################################################
        cluster_algo: ClusterAlgorithm = KMeansClusterAlgorithm,
        cluster_algo_parameters = {},
        # rebuild_cluster: ClusterAlgorithm = KMeansClusterAlgorithm,
        # rebuild_cluster_parameters = {},
        ###########################################################
        superinstance_builder: SuperInstanceBuilder = KMeans_SuperinstanceBuilder(),
        split_superinstance_selection_heur: SuperinstanceSelectionHeuristic = None,
        splitlevel_strategy=None,
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
        metricLearner = None, # wordt momenteel niet gebruikt OBSOLETE
        metricLearer_arguments = {},
        metricLevel = "all",

        ##########
        # During #
        ##########
        metricAmountQueriesAsked = 50,
        learnAMetric = False,

        ###########
        # Initial #
        ########### OBSOLETE
        initial = False,
        initialSupervised = 0, # is een percentage
        initialSemisupervised = 0,
        initialRandom = True, # if false and supervised, gebruik supervised info na initialsemisupervised constraints

        ###########
        # Rebuild #
        ###########
        rebuildPhase = False,
        rebuildLevel = "all", # different levels are all, cluster, superinstance
        # pas belangrijk als er voor verfijningslevel superinstance is gekozen
        rebuildSuperInstanceLevel = 0, # nul is enkel naar de superinstances apart kijken, vanaf ! gaan we van top-down naar beneden kijken

        rebuildAmountQueriesAsked = 50, # vanaf hoeveel queries gevraagd voeren we dit uit  

        rebuilder = SemiCluster,
        rebuildAllOptions = True, # alle gelabellde instances binnen en verfijning mogen worden gebruikt

        rebuildMetric = False,
        rebuilderKeepTransformed = False,

        

        # partition: OBSOLETE
        rebuildPartition = False,
        rebuildPartitionDecider = "all", # verschillede manieren om kleine deeltjes te verplaatsen, all, repres, vote
        rebuildPartitionUseTransformed = False,
        rebuildPartitionLevel = 4,


        #########
        # After #
        #########
        keepSupervised = True, # hebben we standaard op True gezet
        after = False,
        afterAmountQueriesAsked = 50,
        afterMetric = False,
        afterLevel = "all",
        afterSuperInstanceLevel = 0,
        afterAllOptions = True,

        # metric: MetricLearningAlgorithm = EuclidianDistance, # gaan ervanuit dat de caller deze classes al heet initilised
        # metric_parameters = {},
        # rebuilder: InstanceRebuilder = None,
        # rebuilder_parameters = {},
        # baseline = False,

        # ###########
        # # Logging #
        # ########### -> TODO

        # logExtraInfo = False


    ):

        self.seed = seed # seed is belangrijk!!!!!!!!!!!!!!!!!!!!!  

        # init data, querier, max_questions, train_indices and store_intermediate results
        # already initialised so object size does not change during execution
        # python can optimize
        self.data = None
        self.querier = None
        self.train_indices = None

        # init cobras_cluster_algo
        self.cluster_algo = cluster_algo(**cluster_algo_parameters)
        self.superinstance_builder = superinstance_builder
        self.splitlevel_cluster = KMeansClusterAlgorithm(askExtraConstraints=False) # SPLITCLUSTER CLUSTERING ALGO, die werkt op een anderes manier
        ###################
        # METRIC LEARNING #
        ###################
        self.metricLearner = metricLearner # OBSOLETE
        self.metricLearer_arguments = metricLearer_arguments
        self.metricLevel = metricLevel

        ##########
        # During #
        ##########
        self.learnAMetric = learnAMetric
        self.metricAmountQueriesAsked = metricAmountQueriesAsked


        ###########
        # Initial #
        ###########
        self.initial = initial
        self.initialSupervised = initialSupervised # is een percentage
        self.initialSemisupervised = initialSemisupervised
        self.initialRandom = initialRandom

        ###########
        # Rebuild #
        ###########
        self.rebuildPhase = rebuildPhase
        self.rebuildLevel = rebuildLevel # different levels are all, cluster, superinstance
        # pas belangrijk als er voor verfijningslevel superinstance is gekozen
        self.rebuildSuperInstanceLevel = rebuildSuperInstanceLevel # nul is enkel naar de superinstances apart kijken, vanaf ! gaan we van top-down naar beneden kijken

        self.rebuildAmountQueriesAsked = rebuildAmountQueriesAsked # vanaf hoeveel queries gevraagd voeren we dit uit 

        self.rebuilder: Rebuilder = rebuilder()
        self.rebuildAllOptions = rebuildAllOptions

        self.rebuildMetric = rebuildMetric
        self.rebuilderKeepTransformed = rebuilderKeepTransformed
        
        
        


        # OBSOLETE
        self.rebuildPartition = rebuildPartition
        self.rebuildPartitionDecider = rebuildPartitionDecider
        self.rebuildPartitionUseTransformed = rebuildPartitionUseTransformed
        self.rebuildPartitionLevel = rebuildPartitionLevel

        #########
        # After #
        #########
        self.doAfter = after
        self.keepSupervised = keepSupervised
        self.afterAmountQueriesAsked = afterAmountQueriesAsked
        self.afterMetric = afterMetric
        self.afterLevel = afterLevel
        self.afterSuperInstanceLevel = afterSuperInstanceLevel
        self.afterAllOptions = afterAllOptions


        ###################
        # METRIC LEARNING #
        ###################
        # logging
        # self.logExtraInfo = logExtraInfo # wordt momenteel niet gebruikt, nog niet mooi gefixt gekregen TODO

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

        self.certainty_threshold = certainty_threshold

        self.correct_noise = correct_noise

    # LOGGER HERE!
    @property
    def clustering_logger(self):
        return self._cobras_log   
    

    def tokenFit(self, X, nb_clusters, train_indices, querier):
        self.random_generator = np.random.default_rng(self.seed) # hier random_generator gezet
        self._cobras_log.log_start_clustering()
        self.data = X


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




        # during this iteration store the current clustering
        self._cobras_log.update_clustering_to_store(*self.after(), self.keepSupervised)
        self.clustering_to_store = self.clustering.construct_cluster_labeling()



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

            # rebuild phase
            self.rebuild() # dit past de clustering potentieel aan


            # during this iteration store the current clustering
            self._cobras_log.update_clustering_to_store(*self.after(), self.keepSupervised)
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
                self._cobras_log.update_last_intermediate_result(*self.after(), self.keepSupervised)

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
        clusters = self.splitlevel_cluster.cluster( # ML and CL meegeven
            np.copy(self.data), si.indices, k, [], [], seed=self.random_generator.integers(1,1000000), cobras = self # seed voor clusteren wordt rangom gegenereerd (zo krijgen we altijd dezelfde resultaten) => zo moet seed gedaan worden (ook als we ITML enzo oproepen)

        ) if determination else self.cluster_algo.cluster( # ML and CL meegeven
            np.copy(self.data), si.indices, k, [], [], seed=self.random_generator.integers(1,1000000), cobras = self # seed voor clusteren wordt rangom gegenereerd (zo krijgen we altijd dezelfde resultaten) => zo moet seed gedaan worden (ook als we ITML enzo oproepen)

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
                training.append(self.create_superinstance(cur_indices, si)) # hier wordt de parent gezet
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

    def create_superinstance(self, indices, parent=None) -> SuperInstance:
        return self.superinstance_builder.makeSuperInstance(
            self.data, indices, self.train_indices, parent
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
        if self.correct_noise:
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
    
    #######################
    # QUERY RANDOM POINTS #
    ####################### - worden toegevoegd aan de reeds gevraagde constraints TODO OBSOLETE
    def query_random_points(self, options = None, count = 0):
        print("asking random constraints")
        opt = np.array(options).tolist() if options is not None else np.arange(len(self.data)).tolist()
        if self.gen is None:
            self.gen = self.pair_generator(opt) 
        pairs = []
        labels = []
 
        # Get 10 pairs: 
        for i in range(count):
            if self.querier.query_limit_reached():
                print("did not have enough")
                return np.array(pairs), np.array(labels)
            pair = next(self.gen)
            
            while True:
                if self.check_constraint_reuse_between_instances(pair[0], pair[1]) is None: # gaan er ffkes vanuit dat elke superinstance minstens zoveel nodig heeft
                    break
                else: pair = next(self.gen) # niks reusen
            pairs.append(pair)
            cstr = self.query_querier(pair[0], pair[1], "random_points")
            if cstr.is_ML():
                labels.append(1)
            else:
                labels.append(-1)
        return np.array(pairs), np.array(labels)


    def pair_generator(self, numbers): 
        """Return an iterator of random pairs from a list of numbers.""" 
        # Keep track of already generated pairs 
        random.seed(self.seed) # dezelfde seed gebruiken
        used_pairs = set() 
        
        while True: 
            pair = random.sample(numbers, 2) 
            # Avoid generating both (1, 2) and (2, 1) 
            pair = tuple(sorted(pair)) 
            if pair not in used_pairs: 
                used_pairs.add(pair) 
                yield pair

    ################################
    # Get all labelled information #
    ################################

    def getAllLabelled(self):
        clust, r = self.clustering.construct_cluster_labeling(), self.clustering.get_superinstances()
        repres = [s.get_representative_idx() for s in r]

        clust = np.array(clust)

        for blob in self._cobras_log.blobs: # staan momenteel gestored in de logger
            for elem in repres:
                if elem in blob:                
                    repres.extend(blob)
                    clust[np.array(blob)] = clust[elem]
                    break


        repres = list(set(repres)) # hiervan hebben we labelled information, list(set()) is omdat representatives er anders dubeel inzitten

        return np.array(repres), np.array(clust)

    ##################
    # Learn a metric # 
    ##################

    def learnMetric(self): # momenteel steken we alles in de grote data, er is toch geen overlap => is los het gemakkelijkste
        levels = self.getFinegrainedLevel(self.metricLevel)
        data = np.copy(self.data)

        labelled, clust = self.getAllLabelled()

        for level in levels:
            indices = []
            repres = []
            for superinstance in level:
                for idx in superinstance.indices:
                    indices.append(idx)
                    if idx in labelled:
                        repres.append(idx)
            data[np.array(indices)] = LMNN_wrapper(preprocessor = np.copy(self.data), seed = self.seed).fit_transform(None, None, np.copy(repres), clust[np.array(repres)])[np.array(indices)]

        return data



    ############################################
    # get superinstances per finegrained level #
    ############################################
    def getFinegrainedLevel(self,level:str):
        if level == "all":
            return [self.clustering.get_superinstances()]
        if level == "cluster": 
            return self.clustering.get_superinstances_per_cluster()
        if level == "superinstance":
            if self.rebuildSuperInstanceLevel == 0:
                return [[x] for x in self.clustering.get_superinstances()]
            
            supers = [self.clustering.get_superinstances()[0].get_root()]
            for i in range(self.rebuildSuperInstanceLevel):
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
    ########### TODO: het werkt zo niet meer hehe
    def initial_transform(self):
        if not self.initial:
            return
        pairs = None
        y = None
        points = None
        labels = None

        if self.initialRandom:  
            pairs, y = self.querier.getRandomConstraints(self.initialSemisupervised, self.seed)
            points, labels =  self.querier.getRandomLabels(math.ceil(self.initialSupervised*len(self.data)), self.seed)
        else:
            querier = self.querier.createCopy(self.initialSemisupervised) # gaan ervan uit dat dit groter dan nul is
            clusterer = COBRAS(correct_noise=False, seed=self.seed)
            clusterer.fit(np.copy(self.data), -1, None, querier)

            pairs, y = clusterer.constraint_index.getLearningConstraints()

            points, _ = clusterer.getAllLabelled() # TODO: nog fixen
            labels = clusterer.querier.getConstraints(points) # is dit zo hmmmm, moeten we testen
            
        self.data = self.metricLearner(preprocessor = np.copy(self.data), **self.metricLearer_arguments, seed = self.seed).fit_transform(np.copy(pairs), np.copy(y), np.copy(points), np.copy(labels))
        
    ##########
    # During # 
    ########## TODO: nog in de code zetten
    def learnMetricDuring(self):
        if self.learnAMetric and len(self._cobras_log.all_user_constraints) < self.metricAmountQueriesAsked:
            self.data = self.learnMetric()
            self.learnAMetric = False # momenteel doen we het zo, is ook zo bij rebuild, gewoon een keer uitvoeren

    ##############
    # Rebuilding #
    ##############
       

    def rebuild(self):
        if not self.rebuildPhase or len(self._cobras_log.all_user_constraints) < self.rebuildAmountQueriesAsked:
            return
        levels = self.getFinegrainedLevel(self.rebuildLevel)

        labelled, clust = self.getAllLabelled()

        clusters = [Cluster([]) for i in np.unique(clust)] # de nieuwe clusters, is evenveel als dat er labels zijn

        data = self.learnMetric() if self.rebuildMetric else self.data
        
        if self.rebuilderKeepTransformed: # werk later verder met deze transformatie
            self.data = data

        for level in levels:

            if self.rebuildAllOptions: # alles mag gebruikt worden
            # nu begint het echte werk
                indices = []
                repres = []

                for superinstance in level:
                    for idx in superinstance.indices:
                        indices.append(idx)
                        if idx in labelled:
                            repres.append(idx)

                labels_repres = self.rebuilder.rebuild(repres, indices, data, represLabels=np.array(clust)[np.array(repres)]) 

            else: # nu moet het per superinsatance worden gedaan
                indices = []
                repres = [] # alles waar rond kan gebouwd worden
                labelled_instances = [] # hier kan sws alles naartoe
                
                for superinstance in level:
                     for idx in superinstance.indices: # TODO nog nakijken!
                        indices.append(idx)
                        if idx in labelled:
                            repres.append(idx)
                            if idx != repres.representative_idx:
                                labelled_instances.append(idx)

                labelled_instances.append(0) # is voor de representative van de superinstances
                labels_repres = []
                for superinstance in level:
                    labelled_instances[-1] = repres.representative_idx # dit is ook een optie
                    local_labels_repres = self.rebuilder.rebuild(labelled_instances, indices, data, represLabels=np.array(clust)[np.array(labelled_instances)]) # nu enkel labelled instances

                    # nu nog omvormen naar de juiste labels
                    local_labels_repres = np.array(labelled_instances)[local_labels_repres]
                    labels_repres.extend(np.where(np.in1d(repres, local_labels_repres))[0].tolist()) # TODO nog checken!
                                

            # bouw de superinstances en voeg aan de juiste cluster toe
            for i in range(len(indices)):
                idx = indices[i]
                if idx in repres:
                    points = np.array(indices)[labels_repres == labels_repres[i]] # zelfde label als een mogelijke representatieve volgen
                    superinstance = self.superinstance_builder.makeSuperInstance(
                        self.data, points.tolist(), self.train_indices, None
                    )
                    superinstance.representative_idx = idx # manueeel repres zetten
                    clusters[clust[idx]].super_instances.append(superinstance) # 
            


        self.rebuildPhase = False
        self.clustering.clusters = clusters
        # print("done")


    #########
    # AFTER #
    #########
    # def after(self):

    # What to do afterwards, heeft een default TODO
    def after(self):
        
        if not self.doAfter or len(self._cobras_log.all_user_constraints) < self.afterAmountQueriesAsked:
            return np.array(self.clustering.construct_cluster_labeling()), self.clustering.get_superinstances()
        
        levels = self.getFinegrainedLevel(self.afterLevel)
        
        data = self.learnMetric() if self.afterMetric else self.data
        
        labelled, clust = self.getAllLabelled()

        new = np.zeros(len(self.data)) # de nieuwe labels

        for level in levels: # TODO: dit nakijken

            if self.afterAllOptions:
                indices = []
                repres = []
                for superinstance in level:
                    for idx in superinstance.indices:
                        indices.append(idx)
                        if idx in labelled:
                            repres.append(idx)
                model = KNeighborsClassifier(n_neighbors=3)
                model.fit(np.array(data)[np.array(repres)], clust[np.array(repres)])
                new[np.array(indices)] = model.predict(np.array(data)[np.array(indices)])

            else:
                repres = []
                labelled_instances = []
                
                for superinstance in level:
                     for idx in superinstance.indices:
                        if idx in labelled:
                            repres.append(idx)
                            if idx != repres.representative_idx:
                                labelled_instances.append(idx)

                labelled_instances.append(0) #is voor de superinstances
                for superinstance in level:
                    labelled_instances[-1] = repres.representative_idx # dit mag er ook bij
                    model = KNeighborsClassifier(n_neighbors=3)
                    model.fit(np.array(data)[np.array(labelled_instances)], clust[np.array(labelled_instances)])
                    new[np.array(superinstance.indices)] = model.predict(np.array(data)[np.array(superinstance.indices)])

        new[np.array(labelled)] = clust[np.array(labelled)] # die zijn geweten en moeten zo blijven
        return new, self.clustering.get_superinstances()

        
        # clust = np.array(clust)
        # founded = 0
        # for blob in self._cobras_log.blobs:
        #     for elem in repres:
        #         if elem in blob:                
        #             clust[np.array(blob)] = clust[elem]
        #             repres.extend(blob)
        #             founded += 1
        #             break

        # # print(f'blobs not used: {len(self._cobras_log.blobs) - founded}')


        # repres = list(set(repres)) # hiervan hebben we labelled information

        # # print("started after")
        
        # target = np.array(clust)
        # pairs, labels = self.constraint_index.getLearningConstraints()


        # if (len(labels) < 100): # pas vanop een bepaald moment proberen verbeteren
        #     return np.array(clust), r
        # if len(np.unique(target[np.array(repres)])) == 1:
        #     return np.array(clust), r
        #     # newD = ITML_wrapper(preprocessor = np.copy(self.data), seed = self.seed).fit_transform(np.copy(pairs), np.copy(labels), np.copy(repres), target[np.array(repres)])
        # else:
        #     # newD = LMNN_wrapper(preprocessor = np.copy(self.data), seed = self.seed).fit_transform(np.copy(pairs), np.copy(labels), np.copy(repres), target[np.array(repres)])
        #     newD = np.copy(self.data)

        # model = KNeighborsClassifier(n_neighbors=3)
        # model.fit(np.array(newD)[np.array(repres)], target[np.array(repres)])
        # new = model.predict(np.array(newD))


        # new[np.array(repres)] = target[np.array(repres)]

        # # new = target

        # return np.array(new), r