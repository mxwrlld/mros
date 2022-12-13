import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import utils.constants as const
from utils.vector_generator import generate_norm_vector


# ==================== Параметры ==================== #
N = 50
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


def parsen(train_vectors_by_classes: np.ndarray, test_sample: np.ndarray, Bs: np.ndarray) -> int:
    # classes = np.array([np.zeros(shape=(2, 1)) for _vector in train_vectors])
    classification_res = np.ndarray(shape=test_sample.shape[1])
    count_of_classes = len(train_vectors_by_classes)
    count_of_train_vectors = np.sum([train_vectors_by_classes[i].shape[1]
                                     for i in range(count_of_classes)])
    f = np.zeros(count_of_classes)
    P = np.zeros(count_of_classes)
    k = 0.25
    for i in range(test_sample.shape[1]):
        x = test_sample[:, i]
        for j in range(count_of_classes):
            vectors_by_class = train_vectors_by_classes[j]
            _class_dim = vectors_by_class.shape[1]  # -- N
            h = _class_dim ** (- k / 2)   # -- (11)
            const = (2 * np.pi) * (h ** (- 2)) * \
                (np.linalg.det(Bs[j]) ** (- 0.5))
            exp_sequence = []
            for x_i in vectors_by_class.T:
                # (10)
                power = (- 0.5 * (h ** (- 2))) * \
                    (x - x_i) @ np.linalg.inv(Bs[j]) @ (x - x_i)
                exp_sequence.append(const * np.exp(power))
            f[j] = np.average(exp_sequence)
            P[j] = _class_dim
        classification_res[i] = np.argmax((P / count_of_train_vectors) * f)

    return classification_res


def sample_2_classified_sample(sample: np.ndarray, labels: np.ndarray) -> dict:
    vectors_by_classes = []
    count_of_classes = np.unique(labels).shape[0]
    for i in range(count_of_classes):
        class_indexes = np.where(labels == i, True, False)
        vectors_by_classes.append(sample[:, class_indexes])
    return vectors_by_classes


def calc_errors(initial_vectors_by_classes, classified_vectors_by_classes):
    error_vectors = []
    for _class in range(len(initial_vectors_by_classes)):
        initial_vectors_by_class = initial_vectors_by_classes[_class].T
        classified_vectors_by_class = classified_vectors_by_classes[_class].T

        for initial_vector in initial_vectors_by_class:
            if initial_vector not in classified_vectors_by_class:
                error_vectors.append(initial_vector)
    error_vectors = np.array(error_vectors)
    return error_vectors.reshape((2, error_vectors.shape[0]))


if __name__ == "__main__":
    config = {
        "generate": False,
        "save": False,
        "checkpoints": [1]
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

        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        painter("Тестовая выборка", test_vectors[2:], lonely=False)
        fig.add_subplot(1, 2, 2)
        painter("Парзен", vectors_by_classes, lonely=False)
        plt.scatter(errors[0, :], errors[1, :], facecolors='none', s=50,
                    edgecolors='purple', label="Wrong classified vectors", alpha=0.7)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()
