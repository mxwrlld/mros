
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from modules.bayes import *
from modules.knn import k_nearest_neighbours
from modules.parsen import parsen
import utils.constants as const
from utils.vector_generator import generate_norm_vector


# ==================== Параметры ==================== #
N = 200
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
    res = dict()
    config = {
        "generate": False,
        "save": False,
        "checkpoints": [1, 2, 3]
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
    if 0 == 0:
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

    # ==================== Парзен ==================== #
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

        # ==================== Демонстрация классификации ==================== #
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
        R = errors.shape[1] / (test_sample.shape[1])
        print(f"Парзен: \n\tОценка суммарной вероятности ошибочной классификации: ", round(R, 4))

        res["Парзен"] = {
            "vectors_by_classes": vectors_by_classes, "errors":  errors
        }

    # ==================== К ближайших соседей ==================== #
    """
    Построить классификатор, основанный на методе К ближайших соседей (для 
    K=1,3,5), используя сгенерированные в п.1 данные как обучающие выборки, а 
    данные из первой лабораторной работы - как тестовые. Оценить эмпирический 
    риск - оценку суммарной вероятности ошибочной классификации
    """
    if 2 in config["checkpoints"]:
        Ks = [1, 3, 5]
        for K in Ks:
            test_sample = np.concatenate(test_vectors[2:], axis=1)
            train_sample = np.concatenate(train_vectors[2:], axis=1)
            train_labels = np.array([
                _class
                for _class in range(len(train_vectors[2:]))
                for _ in range(train_vectors[2:][_class].shape[1])
            ])
            labels = k_nearest_neighbours(
                train_sample, train_labels, test_sample, K)
            vectors_by_classes = sample_2_classified_sample(
                test_sample, labels)
            errors = calc_errors(test_vectors[2:], vectors_by_classes)

            # ==================== Демонстрация классификации ==================== #
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
            R = errors.shape[1] / (test_sample.shape[1])
            print(
                f"KNN_K - {K}: \n\tОценка суммарной вероятности ошибочной классификации: ", round(R, 4))

            res[f"KNN_K - {K}"] = {
                "vectors_by_classes": vectors_by_classes, "errors":  errors
            }

    # ==================== Байес ==================== #
    """
    Сравнить полученные в пп.2-3 классификаторы и качество их работы с
    байесовским классификатором из л.р.№2.
    """
    if 3 in config["checkpoints"]:
        test_sample = np.concatenate(test_vectors[2:], axis=1)
        labels = bayes(
            test_sample, [B_1, B_2, B_3], [M_1, M_2, M_3])
        vectors_by_classes = sample_2_classified_sample(test_sample, labels)
        errors = calc_errors(test_vectors[2:], vectors_by_classes)

        # ==================== Демонстрация классификации ==================== #
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        painter("Тестовая выборка", test_vectors[2:], lonely=False)
        fig.add_subplot(1, 2, 2)
        painter("Байес", vectors_by_classes, lonely=False)
        plt.scatter(errors[0, :], errors[1, :], facecolors='none', s=27,
                    edgecolors='black', label="Неверная классификация", alpha=1, linewidths=1.5)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.legend()
        plt.show()

        # Оценка эмпирического риска
        R = errors.shape[1] / (test_sample.shape[1])
        print(f"Байес: \n\tОценка суммарной вероятности ошибочной классификации: ", round(R, 4))

        res["Байес"] = {
            "vectors_by_classes": vectors_by_classes, "errors":  errors
        }

    # ==================== Сравнительная демонстрация ==================== #
    if [1, 2, 3] == config["checkpoints"]:
        fig = plt.figure()
        fig.add_subplot(1, 4, 1)
        painter("Тестовая выборка", test_vectors[2:], lonely=False)
        fig.add_subplot(1, 4, 2)
        name = "Парзен"
        painter(name, res[name]
                ["vectors_by_classes"], lonely=False)
        plt.scatter(res[name]["errors"][0, :], res[name]["errors"][1, :], facecolors='none', s=27,
                    edgecolors='black', alpha=1, linewidths=1.5)
        fig.add_subplot(1, 4, 3)
        name = "KNN_K - 5"
        painter(name, res[name]
                ["vectors_by_classes"], lonely=False)
        plt.scatter(res[name]["errors"][0, :], res[name]["errors"][1, :], facecolors='none', s=27,
                    edgecolors='black', alpha=1, linewidths=1.5)
        fig.add_subplot(1, 4, 4)
        name = "Байес"
        painter(name, res[name]
                ["vectors_by_classes"], lonely=False)
        plt.scatter(res[name]["errors"][0, :], res[name]["errors"][1, :], facecolors='none', s=27,
                    edgecolors='black', alpha=1, linewidths=1.5)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.show()
