import numpy as np
from utils.helpers import *


def kmeans_clustering(sample: np.ndarray, k: int, right_clusterization: bool = True):
    if right_clusterization:
        centers = sample[:, sample.shape[1] - k:]
        centers_indexes = [i for i in range(
            sample.shape[1] - k, sample.shape[1])]
    else:
        centers = sample[:, 0:k]
        centers_indexes = [i for i in range(k)]
    count_of_iter = 2
    count_of_changed_class_vectors = []
    min_distances_indexes = None

    while True:
        distances = np.array(
            [calc_euclidean_distances(sample, centers[:, i])
             for i in range(centers.shape[1])],
            dtype=float
        )
        new_min_distances_indexes = np.argmin(distances, axis=0)
        classes_vectors = sample_2_clustered_sample(
            sample, new_min_distances_indexes, k)
        if count_of_iter > 2:
            changed = np.where(min_distances_indexes ==
                               new_min_distances_indexes, 0, 1)
            count_of_changed_class_vectors.append(np.sum(changed))
            # Определяются новые центры кластеров
        i = 0
        new_centers = np.copy(centers)
        for _class in classes_vectors:
            new_centers[:, i] = np.average(classes_vectors[_class], axis=1)
            i += 1

        if np.array_equal(centers, new_centers):
            break
        count_of_iter += 1
        centers = new_centers
        min_distances_indexes = new_min_distances_indexes

    dependency_graph = {
        "iters_count": count_of_iter,
        "count_of_changed_class_vectors": count_of_changed_class_vectors
    }
    clustering_result = {
        "labels": new_min_distances_indexes,
        "centers": centers
    }
    return clustering_result, dependency_graph
