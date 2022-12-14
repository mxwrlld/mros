import collections
import numpy as np
from numpy import linalg as la


def calc_euclidean_distance(x: np.ndarray, y: np.ndarray):
    return np.linalg.norm(x - y)


def knn_classification(x, train_sample, train_labels, K, vcalc_distance):
    distances = vcalc_distance(train_sample.T, x.T)
    neighbors_indexes = distances.argpartition(K)[:K]
    neighbors_classes = train_labels[neighbors_indexes]
    return collections.Counter(neighbors_classes).most_common(1)[0][0]


def k_nearest_neighbours(train_sample: np.ndarray, train_labels: np.ndarray, test_sample: np.ndarray, K: int):
    classification_res = np.ndarray(shape=test_sample.shape[1])

    vcalc_distance = np.vectorize(
        calc_euclidean_distance,
        signature='(n), (m) -> ()'
    )
    vknn_classification = np.vectorize(
        knn_classification,
        signature='(n), (), () -> ()',
        excluded=[1, 2]
    )

    classification_res = vknn_classification(
        test_sample.T, train_sample, train_labels, K, vcalc_distance)
    return classification_res
