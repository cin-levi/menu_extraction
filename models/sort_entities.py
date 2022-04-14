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


def vertical_merge(clusters):
    # sort cluster by x0
    x_0 = [[min([e['box'][0] for e in cluster])] for cluster in clusters]
    dbscan = DBSCAN(eps=2, min_samples=1)
    vertical_cluster_labels = dbscan.fit_predict(x_0)
    unique_clusters = sorted(list(set(vertical_cluster_labels)))
    vertical_merged_clusters = []
    print(unique_clusters)
    print(len(clusters))
    for unique_cluster in unique_clusters:
        _clusters = [clusters[i] for i in range(len(clusters)) if vertical_cluster_labels[i] == unique_cluster]
        _clusters = sorted(_clusters, key=lambda x: min([e['box'][0] for e in x]))
        pointers = list(range(len(_clusters)))
        for i in range(1, len(_clusters) - 1):
            current_labels = [e['entity_type'] for e in _clusters[i]]
            previous_labels = [e['entity_type'] for e in _clusters[i - 1]]
            next_labels = [e['entity_type'] for e in _clusters[i + 1]]
            candidates = []
            if len(set(current_labels + previous_labels)) > len(set(current_labels)):
                candidates.append(i - 1)
            if len(set(current_labels + next_labels)) > len(set(current_labels)):
                candidates.append(i + 1)
            previous_distance = min([e['box'][-1] for e in _clusters[i]]) - min([e['box'][-1] for e in _clusters[i - 1]])
            nex_distance = min([e['box'][-1] for e in _clusters[i + 1]]) - min([e['box'][-1] for e in _clusters[i]])
            if previous_distance < nex_distance:
                pointers[i] = i - 1
            else:
                pointers[i] = i + 1
        # Remove loop
        for i in range(len(pointers)):
            if pointers[i] != i and pointers[pointers[i]] == i:
                pointers[pointers[i]] = pointers[i]
        # Trace to get the cluster using graph
        labels = [i if pointers[i] == i else None for i in range(len(pointers))]
        while not all([l is not None for l in labels]):
            for i in range(len(labels)):
                labels[i] = labels[pointers[i]]
        unique_labels = sorted(list(set(labels)))
        for label in unique_labels:
            _vertical_clusters = []
            for i in range(len(_clusters)):
                if labels[i] == label:
                    _vertical_clusters.extend(_clusters[i])
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
            clusters = sort_entities(data[page])
            for cluster in clusters:
                for entity in cluster:
                    print(f"{entity['entity_type']}: {entity['text']}")
                print("---")
    pass
