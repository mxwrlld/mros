import numpy as np
from utils.helpers import *


def calc_typical_distance(centers: np.ndarray):
    part = 0.5
    distances = [calc_euclidean_distance(centers[:, i], centers[:, j])
                 for i in range(centers.shape[1])
                 for j in range(i, centers.shape[1])
                 if i != j]

    L = centers.shape[1]
    return part * np.sum(distances) * (2 / ((L)*(L - 1)))


def get_first_center(sample: np.ndarray):
    # Нахождение центра первого кластера (C_0),
    # как наиболее удалённого вектора от среднего всех векторов
    average = np.average(sample, axis=1)
    distances = calc_euclidean_distances(sample, average)
    C_0_index = np.argmax(distances)
    C_0 = sample[:, C_0_index]
    return C_0, C_0_index


def get_second_center(sample: np.ndarray, C_0: np.ndarray):
    # Нахождение центра второго кластера (C_1),
    # как наиболее удалённого вектора от центра первого кластера
    # distances = calc_distances(sample, C_0)
    distances = [calc_euclidean_distance(
        C_0, sample[:, i]) for i in range(sample.shape[1])]
    C_1_index = np.argmax(distances)
    C_1 = sample[:, C_1_index]
    return C_1, C_1_index


def maximin_clustering(sample: np.ndarray):
    C_0, C_0_index = get_first_center(sample)
    C_1, C_1_index = get_second_center(sample, C_0)

    centers = np.array([C_0, C_1]).T
    centers_indexes = [C_0_index, C_1_index]
    iters_labels = dict()
    maximin_distances = []
    typical_distances = []

    count_of_iter = 2
    while True:
        distances = np.array(
            [calc_euclidean_distances(sample, centers[:, i], centers_indexes)
             for i in range(centers.shape[1])],
            dtype=float
        )
        min_distances = np.min(distances, axis=0)
        min_distances_indexes = np.argmin(distances, axis=0)
        iters_labels[count_of_iter] = {
            "labels": min_distances_indexes,
            "centers": centers
        }
        C_l_index = np.argmax(min_distances)
        C_l = sample[:, C_l_index].reshape((2, 1))

        # min_c = np.min(calc_euclidean_distances(centers, C_l))
        maximin_distances.append(min_distances[C_l_index])
        typical_distance = calc_typical_distance(centers)
        typical_distances.append(typical_distance)
        if min_distances[C_l_index] <= typical_distance:
            break
        centers = np.append(centers, C_l, axis=1)
        centers_indexes.append(C_l_index)
        count_of_iter += 1

    dependency_graph = {
        "iters_count": count_of_iter + 1,
        "maximin_distances": maximin_distances,
        "typical_distances": typical_distances
    }
    return iters_labels, dependency_graph
