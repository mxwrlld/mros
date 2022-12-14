import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import utils.constants as const
from utils.vector_generator import generate_norm_vector


# ==================== Параметры ==================== #
N = 100
count_of_vectors = 5
B_1, B_2, B_3 = const.B_1, const.B_2, const.B_3
M_1, M_2, M_3 = const.M_1, const.M_2, const.M_3
xs = np.linspace(-3, 3, N)
save_load_path = "data\\5"


def get_vectors(
        count_of_vectors: int, Bs: list, Ms: list,
        generate: bool, save: bool, save_load_path: str, vectors_name: str = "X"):
    if len(Bs) != count_of_vectors or len(Ms) != count_of_vectors:
        raise ValueError(
            'Количество корреляционных матриц и векторов м.о. не совпадает с числом генерируемых векторов')
    vectors = []
    if generate:
        vectors = [generate_norm_vector(N, Bs[i], Ms[i])
                   for i in range(count_of_vectors)]
        if save:
            [np.savetxt(f"{save_load_path}/{vectors_name}_{i + 1}.txt", vectors[i])
             for i in range(count_of_vectors)]
    else:
        vectors = [np.loadtxt(f"{save_load_path}/{vectors_name}_{i + 1}.txt")
                   for i in range(count_of_vectors)]
    return vectors


def painter(title: str, xs: list, lonely: bool = True):
    if lonely:
        plt.figure()
    plt.title(title)
    # "green", "red", "magenta", "pink", "black", "yellow", "cyan"
    cloud_colors = ["blue", "orange", "green", "red", "yellow"]
    for i in range(len(xs)):
        plt.plot(xs[i][0], xs[i][1], c=cloud_colors[i],
                 marker='.', linestyle='none')
    # mng = plt.get_current_fig_manager()
    # mng.window.state('zoomed')
    if lonely:
        plt.show()


def calc_parsen_kernel(x: np.ndarray, x_i: np.ndarray, B: np.ndarray, e_power_multiplier, const):
    power = e_power_multiplier * (x - x_i) @ np.linalg.inv(B) @ (x - x_i)
    return const * np.exp(power)


def parsen_classification(x, train_vectors_by_classes, Bs, count_of_classes, count_of_train_vectors, vcalc_parsen_kernel):
    f = np.zeros(count_of_classes)
    P = np.zeros(count_of_classes)
    k = 0.3

    for j in range(count_of_classes):
        vectors_by_class = train_vectors_by_classes[j]
        _class_dim = vectors_by_class.shape[1]  # -- N
        h = _class_dim ** (- k / 2)   # -- (11)
        const = (2 * np.pi) * (h ** (- 2)) * \
            (np.linalg.det(Bs[j]) ** (- 0.5))
        exp_sequence = []
        e_power_multiplier = (- 0.5 * (h ** (- 2)))
        exp_sequence = vcalc_parsen_kernel(
            x, vectors_by_class.T, Bs[j], e_power_multiplier, const)
        f[j] = np.average(exp_sequence)
        P[j] = _class_dim

    return np.argmax((P / count_of_train_vectors) * f)


def parsen(train_vectors_by_classes: np.ndarray, test_sample: np.ndarray, Bs: np.ndarray):
    classification_res = np.ndarray(shape=test_sample.shape[1])
    count_of_classes = len(train_vectors_by_classes)
    count_of_train_vectors = np.sum([train_vectors_by_classes[i].shape[1]
                                     for i in range(count_of_classes)])

    vcalc_parsen_kernel = np.vectorize(
        calc_parsen_kernel,
        signature='(n), (), ()->()',
        excluded=[0, 2]
    )
    vparsen_classification = np.vectorize(
        parsen_classification,
        signature='(n), (), (), ()->()',
        excluded=[1, 2]
    )

    classification_res = vparsen_classification(
        test_sample.T, train_vectors_by_classes, Bs, count_of_classes, count_of_train_vectors, vcalc_parsen_kernel)

    return classification_res


def calc_euclidean_distance(x: np.ndarray, y: np.ndarray):
    return np.linalg.norm(x - y)


def knn_classification(x, train_sample, train_labels, K, vcalc_distance):
    distances = vcalc_distance(train_sample.T, x.T)
    neighbors_indexes = distances.argpartition(K)[:K]
    neighbors_classes = train_labels[neighbors_indexes]
    return collections.Counter(neighbors_classes).most_common(1)[0][0]


def k_nearest_neighbour(train_sample: np.ndarray, train_labels: np.ndarray, test_sample: np.ndarray, K: int):
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


def sample_2_classified_sample(sample: np.ndarray, labels: np.ndarray) -> dict:
    vectors_by_classes = []
    count_of_classes = np.unique(labels).shape[0]
    for i in range(count_of_classes):
        class_indexes = np.where(labels == i, True, False)
        vectors_by_classes.append(sample[:, class_indexes])
    return vectors_by_classes


def calc_errors(initial_vectors_by_classes, classified_vectors_by_classes):
    error_vectors = None
    vis_classif_error_vector = np.vectorize(
        lambda initial_vector, classified_vectors: (
            initial_vector == classified_vectors).any() == False,
        signature='(n)->()',
        excluded=[1]
    )

    for _class in range(len(initial_vectors_by_classes)):
        initial_vectors_by_class = initial_vectors_by_classes[_class].T
        classified_vectors_by_class = classified_vectors_by_classes[_class].T

        error_vectors_indexes = vis_classif_error_vector(initial_vectors_by_class,
                                                         classified_vectors_by_class)
        if error_vectors is None:
            error_vectors = initial_vectors_by_classes[_class][:,
                                                               error_vectors_indexes]
        else:
            error_vectors = np.concatenate(
                [error_vectors, initial_vectors_by_classes[_class][:, error_vectors_indexes]], axis=1)

    return error_vectors


if __name__ == "__main__":
    config = {
        "generate": True,
        "save": False,
        "checkpoints": [2]
    }

    train_vectors = get_vectors(
        count_of_vectors, [B_1, B_1, B_1, B_2, B_3], [M_1, M_2, M_1, M_2, M_3],
        config["generate"], config["save"], save_load_path, vectors_name="X_train"
    )
    test_vectors = get_vectors(
        count_of_vectors, [B_1, B_1, B_1, B_2, B_3], [M_1, M_2, M_1, M_2, M_3],
        config["generate"], config["save"], save_load_path, vectors_name="X_test"
    )

    # ==================== Демонстрация выборок  ==================== #
    if 0 == 1:
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        painter("Обучающие облачка (разные корр. матрицы)",
                train_vectors[2:], lonely=False)
        fig.add_subplot(1, 2, 2)
        painter("Тестовые облачка (разные корр. матрицы)",
                test_vectors[2:], lonely=False)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()

    # ==================== Парзен  ==================== #
    """
    Построить классификатор, основанный на непараметрической оценки Парзена,
    используя сгенерированные в п.1 данные как обучающие выборки, а данные из
    первой лабораторной работы - как тестовые. В качестве ядра взять гауссовское
    (10), величину h взять в виде (11). Оценить эмпирический риск - оценку
    суммарной вероятности ошибочной классификации
    """
    if 1 in config["checkpoints"]:
        test_sample = np.concatenate(test_vectors[2:], axis=1)
        labels = parsen(train_vectors[2:], test_sample, [B_1, B_2, B_3])
        vectors_by_classes = sample_2_classified_sample(test_sample, labels)
        errors = calc_errors(test_vectors[2:], vectors_by_classes)

        # ==================== Демонстрация классификации  ==================== #
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        painter("Тестовая выборка", test_vectors[2:], lonely=False)
        fig.add_subplot(1, 2, 2)
        painter("Парзен", vectors_by_classes, lonely=False)
        plt.scatter(errors[0, :], errors[1, :], facecolors='none', s=27,
                    edgecolors='black', label="Неверная классификация", alpha=1, linewidths=1.5)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.legend()
        plt.show()

        # Оценка эмпирического риска
        R = errors.shape[1] / (test_sample.shape[0] * N)
        print(f"Оценка суммарной вероятности ошибочной классификации: ", R)

    # ==================== К ближайших соседей  ==================== #
    """
    Построить классификатор, основанный на методе К ближайших соседей (для 
    K=1,3,5), используя сгенерированные в п.1 данные как обучающие выборки, а 
    данные из первой лабораторной работы - как тестовые. Оценить эмпирический 
    риск - оценку суммарной вероятности ошибочной классификации
    """
    if 2 in config["checkpoints"]:
        K = 3
        test_sample = np.concatenate(test_vectors[2:], axis=1)
        train_sample = np.concatenate(train_vectors[2:], axis=1)
        train_labels = np.array([
            _class
            for _class in range(len(train_vectors[2:]))
            for _ in range(train_vectors[2:][_class].shape[1])
        ])
        labels = k_nearest_neighbour(
            train_sample, train_labels, test_sample, K)
        vectors_by_classes = sample_2_classified_sample(test_sample, labels)
        errors = calc_errors(test_vectors[2:], vectors_by_classes)

        # ==================== Демонстрация классификации  ==================== #
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        painter("Тестовая выборка", test_vectors[2:], lonely=False)
        fig.add_subplot(1, 2, 2)
        painter(f"К ближайших соседей. K - {K}",
                vectors_by_classes, lonely=False)
        plt.scatter(errors[0, :], errors[1, :], facecolors='none', s=27,
                    edgecolors='black', label="Неверная классификация", alpha=1, linewidths=1.5)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.legend()
        plt.show()

        # Оценка эмпирического риска
        R = errors.shape[1] / (test_sample.shape[0] * N)
        print(f"Оценка суммарной вероятности ошибочной классификации: ", R)
