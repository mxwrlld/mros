import numpy as np


def sample_2_clustered_sample(sample: np.ndarray, labels: np.ndarray, count_of_clusters: int) -> dict:
    classes_vectors = dict()
    for i in range(count_of_clusters):
        class_indexes = np.where(labels == i, True, False)
        classes_vectors[str(i)] = sample[:, class_indexes]
    return classes_vectors


def calc_euclidean_distance(x: np.ndarray, y: np.ndarray):
    return np.linalg.norm(x - y)


def calc_euclidean_distances(sample: np.ndarray, y: np.ndarray, centers_indexes: list = []):
    return [calc_euclidean_distance(sample[:, i], y)
            if i not in centers_indexes
            else "-inf"
            for i in range(sample.shape[1])]
