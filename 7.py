import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import utils.constants as const
from utils.vector_generator import generate_norm_vector

# ==================== Параметры ==================== #
N = 50
count_of_vectors = 5
B_1 = const.B_4
M_1, M_2, M_3, M_4, M_5 = const.M_1, const.M_2, const.M_3, const.M_4, const.M_5
xs = np.linspace(-3, 3, N)
save_load_path = "data\\7"
# Результат кластеризатора:
#   - число итераций, массивы типичного расстояния, максиминного расстояния
#   Номер итерации:
#       центры,
#       метки векторов,


def get_vectors(
        num_of_vectors: int, Bs: list, Ms: list,
        generate: bool, save: bool, save_load_path: str):
    if len(Bs) != num_of_vectors or len(Ms) != num_of_vectors:
        raise ValueError(
            'Количество корреляционных матриц и векторов м.о. не совпадает с числом генерируемых векторов')
    vectors = []
    if generate:
        vectors = [generate_norm_vector(N, Bs[i], Ms[i])
                   for i in range(num_of_vectors)]
        if save:
            [np.savetxt(f"{save_load_path}/X_{i + 1}.txt", vectors[i])
             for i in range(num_of_vectors)]
    else:
        vectors = [np.loadtxt(f"{save_load_path}/X_{i + 1}.txt")
                   for i in range(num_of_vectors)]
    return vectors


def painter(title: str, xs: list):
    plt.figure()
    plt.title(title)
    # "green", "red", "magenta", "pink", "black", "yellow", "cyan"
    cloud_colors = ["blue", "orange", "green", "red", "yellow"]
    for i in range(len(xs)):
        plt.plot(xs[i][0], xs[i][1], c=cloud_colors[i],
                 marker='.', linestyle='none')
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    plt.show()


def clusters_painter(classes_vectors: dict, centers: np.ndarray, lonely: bool = True, title: str = ""):
    if lonely:
        plt.figure()
    plt.title(title)
    cloud_colors = ["blue", "orange", "green",
                    "magenta", "yellow", "cyan", "black", "pink"]
    i = 0
    for _class in classes_vectors:
        plt.plot(classes_vectors[_class][0], classes_vectors[_class][1],
                 c=cloud_colors[i], marker='.', linestyle='none', label=f"{_class} class")
        i += 1
    plt.plot(centers[0, :], centers[1, :], c="red", marker='.',
             linestyle='none', label=f"centers")
    plt.legend()
    if lonely:
        plt.show()


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


def dependency_graph_painter(
        title: str,
        iters_count: int,
        name_ys: dict,
        lonely: bool = True):
    if lonely:
        plt.figure()
    plt.title(title)
    for name in name_ys:
        plt.plot(np.arange(3, iters_count + 1),
                 name_ys[name], label=name)
    plt.legend()
    if lonely:
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()


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


if __name__ == "__main__":
    config = {
        "generate": False,
        "save": False,
        "checkpoints": [2]
    }

    vectors = get_vectors(
        count_of_vectors, [B_1, B_1, B_1, B_1, B_1], [M_1, M_2, M_3, M_4, M_5],
        config["generate"], config["save"], save_load_path
    )
    painter("Облачка", vectors)

    sample = np.concatenate(vectors, axis=1)

    if 0 in config["checkpoints"]:
        # ==================== Кластеризация sklearn - KMeans (просто пример) ==================== #
        count_of_clusters = 5
        kmeans = KMeans(n_clusters=count_of_clusters)
        y_predicted = kmeans.fit_predict(sample.T)
        classes_vectors = sample_2_clustered_sample(
            sample, kmeans.labels_, count_of_clusters)
        centers = kmeans.cluster_centers_.T
        clusters_painter(classes_vectors, centers)

    if 1 in config["checkpoints"]:
        clustering_result, dependency_graph = maximin_clustering(sample)
        for iter in clustering_result:
            classes_vectors = sample_2_clustered_sample(
                sample, clustering_result[iter]["labels"], clustering_result[iter]["centers"].shape[1])
            clusters_painter(
                classes_vectors, clustering_result[iter]["centers"])
        title = "Зависимость типичного и максиминного расстояния от числа кластеров"
        dependency_graph_painter(
            title,
            dependency_graph["iters_count"],
            {
                "максиминное расстояние": dependency_graph["maximin_distances"],
                "типичное расстояние": dependency_graph["typical_distances"]
            }
        )
        fig = plt.figure()
        i = 1
        for iter in clustering_result:
            fig.add_subplot(2, 3, i)
            classes_vectors = sample_2_clustered_sample(
                sample, clustering_result[iter]["labels"], clustering_result[iter]["centers"].shape[1])
            clusters_painter(
                classes_vectors, clustering_result[iter]["centers"], lonely=False)
            i += 1
        fig.add_subplot(2, 3, (5, 6))
        dependency_graph_painter(
            title,
            dependency_graph["iters_count"],
            {
                "максиминное расстояние": dependency_graph["maximin_distances"],
                "типичное расстояние": dependency_graph["typical_distances"]
            },
            lonely=False
        )
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()

    if 2 in config["checkpoints"]:
        title = "Зависимость числа векторов признаков, сменивших номер класса от номера итерации"
        count_of_clusters = 3
        clustering_result, dependency_graph = kmeans_clustering(
            sample, count_of_clusters)
        classes_vectors = sample_2_clustered_sample(
            sample, clustering_result["labels"], count_of_clusters)
        clusters_painter(
            classes_vectors, clustering_result["centers"], title="KMeans. Число кластеров - 3")
        dependency_graph_painter(
            title,
            dependency_graph["iters_count"],
            {"число векторов сменивших класс":
                dependency_graph["count_of_changed_class_vectors"]}
        )

        count_of_clusters = 5
        clustering_result, dependency_graph = kmeans_clustering(
            sample, count_of_clusters)
        classes_vectors = sample_2_clustered_sample(
            sample, clustering_result["labels"], count_of_clusters)
        clusters_painter(
            classes_vectors, clustering_result["centers"], title="KMeans. Число кластеров - 5.\n \"Правильная\" кластеризация")
        dependency_graph_painter(
            title,
            dependency_graph["iters_count"],
            {"число векторов сменивших класс":
                dependency_graph["count_of_changed_class_vectors"]}
        )

        count_of_clusters = 5
        clustering_result, dependency_graph = kmeans_clustering(
            sample, count_of_clusters, right_clusterization=False)
        classes_vectors = sample_2_clustered_sample(
            sample, clustering_result["labels"], count_of_clusters)
        clusters_painter(
            classes_vectors, clustering_result["centers"], title="KMeans. Число кластеров - 5.\n \"Неправильная\" кластеризация"
        )
        dependency_graph_painter(
            title,
            dependency_graph["iters_count"],
            {"число векторов сменивших класс":
                dependency_graph["count_of_changed_class_vectors"]}
        )
