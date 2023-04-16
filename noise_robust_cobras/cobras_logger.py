import copy
import time
from typing import List

import numpy as np

from noise_robust_cobras.noise_robust.datastructures.constraint import Constraint


class NopLogger(object):
    def nop(*args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop


class ClusteringLogger:
    def __init__(self):
        # start time
        self.start_time = None

        # basic logging of intermediate results
        self.intermediate_results = []

        # all constraints obtained from the user
        self.all_user_constraints = []

        # algorithm phases
        self.current_phase = None
        self.algorithm_phases = []

        # (detected) noisy constraints
        self.corrected_constraint_sets = []
        self.noisy_constraint_data = []
        self.detected_noisy_constraint_data = []

        # clustering to store
        self.clustering_to_store = None

        # execution time
        self.execution_time = None

        ###################
        # Metric learning #
        ###################

        self.blobs = [] # dit moet eigenlijk op een andere manier, ma ja

        self.seen_indices = []

        # self.allSeenSuperinstances = set()

        # self.currentSuperinstances = []

        # self.superinstances = []

        # self.currentrepres = []

        # self.repres = [] 

    #########################
    # information retrieval #
    #########################
    
    # mss hier dan ook iets teruggeven van alle tussenliggende metrics die gemaakt zijn

    def get_all_clusterings(self):
        return [cluster for cluster, _, _ in self.intermediate_results]

    def get_runtimes(self):
        return [runtime for _, runtime, _ in self.intermediate_results]

    def get_ml_cl_constraint_lists(self):
        ml = []
        cl = []
        for constraint in self.all_user_constraints:
            if constraint.is_ML():
                ml.append(constraint.get_instance_tuple())
            else:
                cl.append(constraint.get_instance_tuple())
        return ml, cl

    # def getConstraints(self):
    #     pairs = []
    #     labels = []
    #     for constraint in self.all_user_constraints:
    #         pairs.append(constraint.get_instance_tuple())
    #         if constraint.is_ML():
    #            labels.append(1)
    #         else:
    #             labels.append(-1)
    #     return np.c_[ np.array(pairs), np.array(labels) ]
    
    # def getSuperinstances(self):
    #     return np.array(self.superinstances)
    
    # def getRepres(self):
    #     l1=list(range(len(self.repres)))
    #     d1=zip(l1,self.repres)
    #     return dict(d1)

    def add_mistake_information(self, ground_truth_querier):
        for i, (constraint_number, constraint_copy) in enumerate(
            self.corrected_constraint_sets
        ):
            mistakes = []
            for con in constraint_copy:
                if (
                    ground_truth_querier.query(*con.get_instance_tuple()).is_ML()
                    != con.is_ML()
                ):
                    mistakes.append(con)
            self.corrected_constraint_sets[i] = (
                constraint_number,
                constraint_copy,
                mistakes,
            )

    ###################
    # log constraints #
    ###################
    def log_new_user_query(self, constraint): # dit is nagekeken
        # add it to the blob yeet
        if constraint.is_ML():
            ml = constraint.get_instance_tuple()
            ind1 = ml[0]
            ind2 = ml[1]
            blob1 = []
            blob2 = []
            if ind1 in self.seen_indices:
                for blob in self.blobs:
                    if ind1 in blob:
                        blob1 = blob
                        break
            if ind2 in self.seen_indices:
                for blob in self.blobs:
                    if ind2 in blob:
                        blob2 = blob
                        break

            if ind1 in blob2: # dan zitten ze in dezelfde blob
                pass
            elif len(blob1) > 0 and len(blob2) > 0:
                blob1.extend(blob2)
                self.blobs.remove(blob2)
            elif len(blob1) > 0:
                blob1.append(ind2)
                self.seen_indices.append(ind2)
            elif len(blob2) > 0:
                blob2.append(ind1)
                self.seen_indices.append(ind1)
            else:
                self.blobs.append([ind1, ind2])
                self.seen_indices.extend([ind1, ind2])
        # add the constraint to all_user_constraints
        self.all_user_constraints.append(constraint)

        # remember the superinstances and repres
        # self.allSeenSuperinstances.update(self.currentSuperinstances) # houdt de de current superinstances bij
        # self.superinstances.append(self.currentSuperinstances)
        # self.repres.append(self.currentrepres)

        # keep algorithm phases up to date
        self.algorithm_phases.append(self.current_phase)

        # intermediate clustering results
        self.intermediate_results.append(
            (
                self.clustering_to_store,
                time.time() - self.start_time,
                len(self.all_user_constraints),
            )
        )

       

    ##################
    # execution time #
    ##################

    def log_start_clustering(self):
        self.start_time = time.time()

    def log_end_clustering(self):
        self.execution_time = time.time() - self.start_time

    ##############
    # phase data #
    ##############

    def log_entering_phase(self, phase): # ook nuttig
        self.current_phase = phase

    ###############
    # clusterings #
    ###############

    def update_clustering_to_store(self, clustering, superinstances, blobThing = False):
        if isinstance(clustering, np.ndarray):
            self.clustering_to_store = clustering.tolist()
        elif isinstance(clustering, list):
            self.clustering_to_store = list(clustering)
        else:
            self.clustering_to_store = clustering.construct_cluster_labeling()

        currentrepres = []
        # currentSuperinstances = np.zeros(len(self.clustering_to_store))

        for i, super in enumerate(superinstances): # oh wauw
            currentrepres.append(super.get_representative_idx())
            # currentSuperinstances[np.array(super.indices)] = i


        # self.currentrepres = currentrepres
        # self.currentSuperinstances = currentSuperinstances.tolist()

        if blobThing:
            self.clustering_to_store = np.array(self.clustering_to_store)
            for blob in self.blobs:
                for elem in currentrepres:
                    if elem in blob:                
                        self.clustering_to_store[np.array(blob)] = self.clustering_to_store[elem]
                        break
            self.clustering_to_store = self.clustering_to_store.tolist()

        

    def update_last_intermediate_result(self, clustering, superinstances, blobThing = False):
        if len(self.intermediate_results) == 0:
            return
        if not isinstance(clustering, np.ndarray):
            self.intermediate_results[-1] = (
                clustering.construct_cluster_labeling(),
                time.time() - self.start_time,
                len(self.all_user_constraints),
            )
        else:
            self.intermediate_results[-1] = (
                clustering.tolist(),
                time.time() - self.start_time,
                len(self.all_user_constraints),
            )

        currentrepres = []
        currentSuperinstances = np.zeros(len(self.clustering_to_store))

        for i, super in enumerate(superinstances): # oh wauw
            currentrepres.append(super.get_representative_idx())
            # currentSuperinstances[np.array(super.indices)] = i


        # self.repres[-1] = currentrepres
        # self.superinstances[-1] = currentSuperinstances.tolist()

        if blobThing:
            clustering_to_store = np.array(self.intermediate_results[-1][0])
            for blob in self.blobs:
                for elem in currentrepres:
                    if elem in blob:                 
                        clustering_to_store[np.array(blob)] = clustering_to_store[elem]
                        break
            self.intermediate_results[-1] = (
                clustering_to_store.tolist(),
                time.time() - self.start_time,
                len(self.all_user_constraints),
            )
            

    #####################
    # noisy constraints #
    #####################

    def log_corrected_constraint_set(self, constraints):
        constraint_copy: List[Constraint] = [copy.copy(con) for con in constraints]
        current_constraint_number = len(self.all_user_constraints)
        self.corrected_constraint_sets.append(
            (current_constraint_number, constraint_copy)
        )

    def log_detected_noisy_constraints(self, constraints):
        con_length = len(self.all_user_constraints)
        for con in constraints:
            self.detected_noisy_constraint_data.append((con_length, copy.copy(con)))


