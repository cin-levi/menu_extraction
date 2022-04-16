import math

from sklearn.cluster import DBSCAN, KMeans
import numpy as np


def sort_entities(output):
    accepted_entities = []
    other_entities = []
    for key in output:
        for entity in output[key]:
            entity['box'] = [min([x[0] for x in entity['position']]),
                             min([x[1] for x in entity['position']]),
                             max([x[2] for x in entity['position']]),
                             max([x[3] for x in entity['position']]),
                             ]
        accepted_entities.extend([x for x in output[key] if x['entity_type'] in ['v', 'w', 'p']])
        other_entities.extend([x for x in output[key] if x['entity_type'] not in ['v', 'w', 'p']])
    if len(accepted_entities) > 0:
        clusters, _ = sort_parts(accepted_entities)
    else:
        clusters = []
    return clusters, []


def horizontal_merge(entities):
    # TODO: Improve later
    db_scan = DBSCAN(eps=2, min_samples=1)
    y_centers = [(e['box'][-1] + e['box'][1]) / 2 for e in entities]
    y_centers = np.array(y_centers).reshape(-1, 1)
    y_labels = db_scan.fit_predict(y_centers)

    cluster_pointer = [i if entities[i]['entity_type'] == 'w' else -1 for i in range(len(y_labels))]
    for i in range(len(y_labels)):
        if entities[i]['entity_type'] != 'w':
            candidates = [j for j in range(len(entities)) if
                          entities[j]['entity_type'] == 'w' and y_labels[j] == y_labels[i] and
                          entities[j]['box'][0] <= entities[i]['box'][0] + 5]
            if len(candidates) > 0:
                cluster_pointer[i] = candidates[-1]
            else:
                cluster_pointer[i] = i
    clusters = [[entities[i] for i in range(len(entities)) if cluster_pointer[i] == index] for index in
                sorted(list(set(cluster_pointer)))]
    return clusters


def calculate_distance(cluster_1, cluster_2):
    x1 = min([e['box'][0] for e in cluster_1])
    y1 = min([e['box'][1] for e in cluster_1])
    x2 = min([e['box'][0] for e in cluster_2])
    y2 = min([e['box'][1] for e in cluster_2])
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


def vertical_merge(clusters):
    # sort cluster by x0, chose either name or price , the one have less value
    num_prices = len([e for c in clusters for e in c if e['entity_type'] == 'p'])
    num_names = len([e for c in clusters for e in c if e['entity_type'] == 'w'])
    accepted_label = 'p' if num_prices <= num_names else 'w'
    pointer = [i if accepted_label in [e['entity_type'] for e in clusters[i]] else None for i in range(len(clusters))]
    # Find closest
    for i in range(len(pointer)):
        if pointer[i] is None:
            distance = [calculate_distance(clusters[i], clusters[j]) if pointer[j] is not None else 5e10 for j in
                        range(len(clusters))]
            pointer[i] = np.argmin(distance)

    for i in range(len(pointer)):
        if pointer[i] != i and pointer[pointer[i]] == i:
            pointer[pointer[i]] = pointer[i]

    # Trace to get the cluster using graph
    vertical_merged_clusters = []
    labels = [i if pointer[i] == i else None for i in range(len(pointer))]
    while not all([l is not None for l in labels]):
        for i in range(len(labels)):
            labels[i] = labels[pointer[i]]
    unique_labels = sorted(list(set(labels)))

    for label in unique_labels:
        _vertical_clusters = []
        for i in range(len(labels)):
            if labels[i] == label:
                _vertical_clusters.extend(clusters[i])
        vertical_merged_clusters.append(_vertical_clusters)

    return vertical_merged_clusters


def sort_parts(entities):
    """
    The algorithm will based on the position of wine name
    Will detect 1 columns or 2 columns - Later
    """
    clusters = horizontal_merge(entities)
    complete_clusters = [c for c in clusters if all([l in [e['entity_type'] for e in c] for l in ['w', 'v', 'p']])]
    incomplete_clusters = [c for c in clusters if
                           not all([l in [e['entity_type'] for e in c] for l in ['w', 'v', 'p']])]
    vertical_merged_clusters = vertical_merge(incomplete_clusters)
    complete_clusters.extend(vertical_merged_clusters)
    return complete_clusters, None


if __name__ == '__main__':
    import json

    data_file = 'D:\\menu_extraction\\predictions.json'
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for page in data:
        if len(data[page]) > 0:
            clusters, _ = sort_entities(data[page])
            for cluster in clusters:
                for entity in cluster:
                    # print(entity)
                    print(f"{entity['entity_type']}: {entity['text']}")
                print("---")
    pass
