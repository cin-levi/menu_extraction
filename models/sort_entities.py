from sklearn.cluster import DBSCAN, KMeans
import numpy as np


def sort_parts(entities):
    """
    The algorithm will based on the position of wine name
    Will detect 1 columns or 2 columns - Later
    """
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
    return clusters, None
