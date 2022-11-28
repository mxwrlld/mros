import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import utils.constants as const
from utils.vector_generator import generate_norm_vector
from modules.lssvm import LSSVM


# ==================== Параметры ==================== #
N = 200
B_1, B_2 = const.B_1, const.B_2
M_1_ls, M_2_ls = const.M_1_ls, const.M_2_ls
M_1_lis, M_2_lis = const.M_1_lis, const.M_2_lis
xs = np.linspace(-3, 3, N)


def get_vectors(generate: bool, save: bool):
    if generate:
        Y_1 = generate_norm_vector(N, B_1, M_1_ls)
        Y_2 = generate_norm_vector(N, B_1, M_2_ls)
        X_1 = generate_norm_vector(N, B_1, M_1_lis)
        X_2 = generate_norm_vector(N, B_2, M_2_lis)
        if save:
            np.savetxt("4/data/Y_1.txt", Y_1)
            np.savetxt("4/data/Y_2.txt", Y_2)
            np.savetxt("4/data/X_1.txt", X_1)
            np.savetxt("4/data/X_2.txt", X_2)
    else:
        Y_1 = np.loadtxt("data/Y_1.txt")
        Y_2 = np.loadtxt("data/Y_2.txt")
        X_1 = np.loadtxt("data/X_1.txt")
        X_2 = np.loadtxt("data/X_2.txt")
    return Y_1, Y_2, X_1, X_2


def generate_training_sample(x_1: np.ndarray, x_2: np.ndarray):
    length = x_1.shape[1]
    m = x_1.shape[1] + x_2.shape[1]
    zs = np.ndarray(shape=(4, m))
    for i in range(m):
        if i % 2 == 0:
            zs[0, i] = x_1[0, int((i / 2) % length)]
            zs[1, i] = x_1[1, int((i / 2) % length)]
            zs[3, i] = - 1
        else:
            zs[0, i] = x_2[0, int((i - 1 / 2) % length)]
            zs[1, i] = x_2[1, int((i - 1 / 2) % length)]
            zs[3, i] = 1
        zs[2, i] = 1
    # Дополнительное переупорядочивание
    # np.random.default_rng().shuffle(zs, axis=1)
    return zs


def calc_alpha(classify_sequence: np.ndarray, class_type: int):
    return len(classify_sequence[classify_sequence == class_type]) / len(classify_sequence)


def calc_decisive_function(w: np.ndarray, w_n: float, x: np.ndarray):
    return w[0, 0] * x[0] + w[1, 0] * x[1] + w_n


def calc_decisive_functions(w: np.ndarray, w_n: float, xs: np.ndarray):
    v_calc_dec_func = np.vectorize(calc_decisive_function,
                                   excluded=[0], signature='(),(n)->()')
    return v_calc_dec_func(w, w_n, np.reshape(xs.T, newshape=(xs.shape[1], xs.shape[0])))


def calc_decisive_boundary(w: np.ndarray, w_n: float, x: float):
    return (- w[0, 0] * x - w_n) / w[1, 0]


def calc_decisive_boundaries(w: np.ndarray, w_n: float, xs: np.ndarray):
    v_calc_bound = np.vectorize(calc_decisive_boundary, excluded=[0])
    return v_calc_bound(w, w_n, xs)


def classify_vectors(w: np.ndarray, w_n: float, xs: np.ndarray, class_type: int, another_class_type: int):
    ds = calc_decisive_functions(w, w_n, xs)
    if class_type == 0:
        return np.where(ds < 0, class_type, another_class_type)
    return np.where(ds > 0, class_type, another_class_type)


def painter(title: str, xs, x_1, x_2, name_ys: dict, isDiffBayes: bool = False):
    plt.title(title)
    plt.plot(x_1[0], x_1[1], c="blue", marker='.', linestyle='none')
    plt.plot(x_2[0], x_2[1], c="orange", marker='.', linestyle='none')
    colors = ["green", "red", "magenta", "pink", "black", "yellow", "cyan"]
    i = 0
    for name in name_ys:
        plt.plot(xs, name_ys[name]["ys"],
                 c=colors[i], label=name)
        if ('support_vectors' in name_ys[name]) and (name_ys[name]['support_vectors'] is not None):
            plt.plot(name_ys[name]['support_vectors']["class_0"][0, :],
                     name_ys[name]['support_vectors']["class_0"][1, :],
                     c="blue", marker='x', linestyle='none')
            plt.plot(name_ys[name]['support_vectors']["class_1"][0, :],
                     name_ys[name]['support_vectors']["class_1"][1, :],
                     c="orange", marker='x', linestyle='none')
        i += 1
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.xlim([-4, 4])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    res = dict()
    config = {
        "generate": False,
        "save": False
    }
    # ==================== Синтез выборок двух классов ==================== #
    # 1. Синтезировать линейно разделимые выборки для двух классов двумерных случайных векторов в количестве N=100 в каждом классе
    S_1, S_2, IS_1, IS_2 = get_vectors(
        config["generate"], config["save"])

    # ==================== SVM для ЛР классов ==================== #
    # 2. Построить линейный классификатор по методу опорных векторов на выборке с линейно разделимыми классами.
    # Использовать:
    # - задачу (7) и метод решения квадратичных задач
    # - метод sklearn.svm.SVC библиотеки scikit-learn
    # сопоставить решения из п.(б) с решением методом sklearn.svm.LinearSVC
    training_sample_ls = generate_training_sample(S_1, S_2)

    lssvm = LSSVM(training_sample_ls)
    res["LS"] = {
        "lssvm": {
            "support_vectors": {
                "class_0": lssvm.support_vectors[0:2, lssvm.support_vectors[3, :] == -1],
                "class_1": lssvm.support_vectors[0:2, lssvm.support_vectors[3, :] == 1]
            },
            "ys": calc_decisive_boundaries(lssvm.w, lssvm.w_n, xs),
            "p0": calc_alpha(classify_vectors(lssvm.w, lssvm.w_n, S_1, 0, 1), 1),
            "p1": calc_alpha(classify_vectors(lssvm.w, lssvm.w_n, S_2, 1, 0), 0)
        }
    }
    # print(lssvm.support_vectors)
    # print(lssvm.support_vectors[0:2, lssvm.support_vectors[3, :] == 1])
    # print(lssvm.support_vectors[0:2, lssvm.support_vectors[3, :] == -1])

    sklearn_svm = svm.SVC(kernel='linear')
    sklearn_svm.fit(training_sample_ls[0:3, :].T, training_sample_ls[3, :])
    w = sklearn_svm.coef_.T
    w_n = sklearn_svm.intercept_[0]
    support_vectors = training_sample_ls[:, sklearn_svm.support_]
    res["LS"].update({
        "sklearn_svm": {
            "support_vectors": {
                "class_0": support_vectors[0:2, support_vectors[3, :] == -1],
                "class_1": support_vectors[0:2, support_vectors[3, :] == 1]
            },
            "ys": calc_decisive_boundaries(w, w_n, xs),
            "p0": calc_alpha(classify_vectors(w, w_n, S_1, 0, 1), 1),
            "p1": calc_alpha(classify_vectors(w, w_n, S_2, 1, 0), 0)
        }
    })

    sklearn_linearSVC = svm.LinearSVC()
    sklearn_linearSVC.fit(
        training_sample_ls[0:3, :].T, training_sample_ls[3, :])
    w = sklearn_linearSVC.coef_.T
    w_n = sklearn_linearSVC.intercept_[0]
    res["LS"].update({
        "sklearn_linearSVC": {
            "support_vectors": None,
            "ys": calc_decisive_boundaries(w, w_n, xs),
            "p0": calc_alpha(classify_vectors(w, w_n, S_1, 0, 1), 1),
            "p1": calc_alpha(classify_vectors(w, w_n, S_2, 1, 0), 0)
        }
    })

    painter(
        "Линейно разделимые классы", xs, S_1, S_2,
        {"lssvm": res["LS"]["lssvm"]}
    )
    painter(
        "Линейно разделимые классы", xs, S_1, S_2, res["LS"]
    )
    # 3. Построить линейный классификатор по SVM на выборке с линейно неразделимыми классами.
    # Использовать:
    # - задачу (12) и метод решения квадратичных задач.
    #       Указать решения для C=1/10, 1, 10 и подобранно самостоятельно «лучшим коэффициентом».
    # - метод sklearn.svm.SVC библиотеки scikit-learn

    # 4. Построить классификатор по SVM, разделяющий линейно неразделимые классы.
    # Использовать:
    # - задачу (14) и метод решения квадратичных задач,
    #       Исследовать решение для различных значений параметра C=1/10, 1, 10 и различных ядер из таблицы 1
    # - метод sklearn.svm.SVC.

    # z0, z1 = S_1, S_2
    # x = np.concatenate((z0, z1), axis=1).T
    # yldeal = np.zeros(shape=2*N)
    # yldeal[N: 2*N] = 1
    # clf = svm.LinearSVC()
    # clf.fit(x, yldeal)

    # x1, x2 = -3, 3

    # def get_y(x): return -(clf.coef_[0, 0] *
    #                        x + clf.intercept_) / clf.coef_[0, 1]
    # y1, y2 = get_y(x1), get_y(x2)

    # plt.plot(z0[0], z0[1], color='red', marker='.', linestyle='none')
    # plt.plot(z1[0], z1[1], color='blue', marker='.', linestyle='none')
    # plt.plot([x1, x2], [y1, y2], color='green', marker='X', linestyle='solid')
    # plt.show()

    # yPredicted = clf.predict(x)
    # yDif = np.abs(yldeal - yPredicted)
    # Nerr = np.sum(yDif)
    # yDif01 = yDif[0:N]
    # yDif10 = yDif[N:2*N]
    # N01 = np.sum(yDif01)
    # N10 = np.sum(yDif10)
    # print(Nerr/N)
    # print(N01/N)
    # print(N10/N)
