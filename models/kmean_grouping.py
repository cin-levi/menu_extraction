from typing import *
from sklearn.cluster import KMeans
import numpy as np
import operator
import copy
import math
import collections


def normalize_and_mapping_data(all_entities):
    all_consider_entities = sorted([e for e in all_entities], key=lambda x: (x['box'][1], x['box'][0]))
    min_y = min([e['box'][1] for e in all_consider_entities])
    max_x = max([e['box'][2] for e in all_consider_entities])
    # TODO: need to improve
    list_atribute = [(e['box'][1] - min_y) * max_x + e['box'][0] for e in all_consider_entities]
    for i in range(len(all_consider_entities)):
        all_consider_entities[i]['attribute'] = list_atribute[i]
    return all_consider_entities, list_atribute


def calculate_loss(clusters, label_list, group_completed, add_coefficient):
    cluster_labels = list(set(clusters))
    group_completed_value = sum(
        [1 + add_coefficient[label] if label in add_coefficient else 1 for label in group_completed])
    loss = 0
    for cluster_label in cluster_labels:
        a_cluster = [label_list[i] for i in range(len(clusters)) if clusters[i] == cluster_label]
        a_cluster_setted_value = sum(
            [1 + add_coefficient[label] if label in add_coefficient else 1 for label in list(set(a_cluster))])
        incomplet_loss = min(group_completed_value - a_cluster_setted_value, a_cluster_setted_value)
        a_cluster_value = sum([1 + add_coefficient[label] if label in add_coefficient else 1 for label in a_cluster])
        redandunt_loss = a_cluster_value - a_cluster_setted_value
        loss += incomplet_loss + redandunt_loss
    return loss


def sort_by_first_index(final_entity_segmented):
    first_index_to_entity = {min([e['attribute'] for e in part]): part for part in final_entity_segmented}
    first_index_to_entity = collections.OrderedDict(sorted(first_index_to_entity.items()))
    return [first_index_to_entity[e] for e in first_index_to_entity]


def sort_parts(all_entities: List[Dict] = None):
    if len(all_entities) == 0:
        return [], []
    all_consider_entities, list_atribute = normalize_and_mapping_data(all_entities)
    add_coefficient = {'item_name': 1}
    list_atribute = np.array(list_atribute).reshape(-1, 1)
    label_list = [e['entity_type'] for e in all_consider_entities]
    group_completed = list(set([e['entity_type'] for e in all_consider_entities]))
    label_num = {label: sum([1 if e['entity_type'] == label else 0 for e in all_consider_entities]) for label in
                 group_completed}
    max_cluster = max([label_num[l] for l in label_num])
    min_cluster = min([label_num[l] for l in label_num])

    clusters_loss = []
    centers_candidate = []
    clusters_candidate = []
    for i in range(2):
        curr_loss, curr_center, curr_cluster = run_kmean(list_atribute, min_cluster, max_cluster, label_list,
                                                         group_completed, add_coefficient)
        clusters_loss.extend(curr_loss), centers_candidate.extend(curr_center), clusters_candidate.extend(curr_cluster)
    min_loss = min(clusters_loss)
    min_loss_index = max([i for i in range(len(clusters_loss)) if clusters_loss[i] == min_loss])
    final_cluster = clusters_candidate[min_loss_index]
    final_centers = centers_candidate[min_loss_index]

    cluster_labels = list(set(final_cluster))
    entity_segmented = []
    for label in cluster_labels:
        current_culster = [all_consider_entities[i] for i in range(len(all_consider_entities)) if
                           final_cluster[i] == label]
        entity_segmented.append(current_culster)
    final_entity_segmented = sort_by_first_index(entity_segmented)
    return final_entity_segmented, all_entities


# In order to get stable kmean
def run_kmean(list_atribute, min_cluster, max_cluster, label_list, group_completed, add_coefficient):
    clusters_candidate = []
    clusters_loss = []
    centers_candidate = []
    for i in range(min_cluster, max_cluster + 1):
        clustering = KMeans(n_clusters=i).fit(list_atribute)
        clusters = clustering.labels_
        clusters_candidate.append(clusters)
        loss = calculate_loss(clusters=clusters, label_list=label_list, group_completed=group_completed,
                              add_coefficient=add_coefficient)
        clusters_loss.append(loss)
        center = [x[0] for x in clustering.cluster_centers_]
        centers_candidate.append(center)
    return clusters_loss, centers_candidate, clusters_candidate
