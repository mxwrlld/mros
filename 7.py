import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import utils.constants as const
from utils.vector_generator import generate_norm_vector

# ==================== Параметры ==================== #
N = 50
num_of_vectors = 5
B_1 = const.B_4
M_1, M_2, M_3, M_4, M_5 = const.M_1, const.M_2, const.M_3, const.M_4, const.M_5
xs = np.linspace(-3, 3, N)
save_load_path = "data\\7"


def get_vectors(
        num_of_vectors: int, Bs: list, Ms: list,
        generate: bool, save: bool, save_load_path: str):
    if len(Bs) != num_of_vectors or len(Ms) != num_of_vectors:
        raise ValueError('Количество корреляционных матриц и векторов м.о. не совпадает с числом генерируемых векторов')
    vectors = []
    if generate:
        vectors = [generate_norm_vector(N, Bs[i], Ms[i]) for i in range(num_of_vectors)]
        if save: 
            [np.savetxt(f"{save_load_path}/X_{i + 1}.txt", vectors[i]) for i in range(num_of_vectors)]
    else:
        vectors = [np.loadtxt(f"{save_load_path}/X_{i + 1}.txt") for i in range(num_of_vectors)]
    return vectors


def painter(title: str, xs: list):
    plt.figure()
    plt.title(title)
    # "green", "red", "magenta", "pink", "black", "yellow", "cyan"
    cloud_colors = ["blue", "orange", "green", "red", "yellow"]
    for i in range(len(xs)):
        plt.plot(xs[i][0], xs[i][1], c=cloud_colors[i], marker='.', linestyle='none')
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    plt.show()


def clusters_painter(classes_vectors: dict):
    plt.figure()
    cloud_colors = ["blue", "orange", "green", "red", "yellow"]
    i = 0
    for _class in classes_vectors:
        plt.plot(classes_vectors[_class][0], classes_vectors[_class][1], c=cloud_colors[i], marker='.', linestyle='none', label=f"{_class} class")
        i += 1
    plt.legend()
    plt.show()


def sample_2_clustered_sample(sample: np.ndarray, labels: np.ndarray, count_of_clusters: int) -> dict:
    classes_vectors = dict()
    for i in range(count_of_clusters):
        class_indexes = np.where(kmeans.labels_ == i, True, False)
        classes_vectors[str(i)] = sample[:, class_indexes]
    return classes_vectors


if __name__ == "__main__":
    res = dict()
    config = {
        "generate": False,
        "save": False,
        "checkpoints": []
    }

    vectors = get_vectors(
        num_of_vectors, [B_1, B_1, B_1, B_1, B_1], [M_1, M_2, M_3, M_4, M_5], 
        config["generate"], config["save"], save_load_path
    )
    #painter("Облачка", vectors)

    sample = np.concatenate(vectors, axis=1)
    #print(sample.T[0:3, :])

    # ==================== Кластеризация sklearn (просто пример) ==================== #
    count_of_clusters = 5
    kmeans = KMeans(n_clusters=count_of_clusters)
    y_predicted = kmeans.fit_predict(sample.T)
    classes_vectors = sample_2_clustered_sample(sample, kmeans.labels_, count_of_clusters)
    clusters_painter(classes_vectors)
    