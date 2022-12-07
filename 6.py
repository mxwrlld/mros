import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import utils.constants as const
from utils.vector_generator import generate_norm_vector
from modules.svm.lssvm import LSSVM
from modules.svm.lissvm import LISSVM
from modules.svm.klissvm import KLISSVM


# ==================== Параметры ==================== #
N = 100
B_1, B_2 = const.B_1, const.B_2
M_1_ls, M_2_ls = const.M_1_ls, const.M_2_ls
M_1_lis, M_2_lis = const.M_1_lis, const.M_2_lis
xs = np.linspace(-3, 3, N)


def get_vectors(generate: bool, lssave: bool, lissave: bool):
    if generate:
        Y_1 = generate_norm_vector(N, B_1, M_1_ls)
        Y_2 = generate_norm_vector(N, B_2, M_2_ls)
        X_1 = generate_norm_vector(N, B_1, M_1_lis)
        X_2 = generate_norm_vector(N, B_2, M_2_lis)
        if lssave:
            # np.savetxt("data/ls/Y_1.txt", Y_1)
            # np.savetxt("data/ls/Y_2.txt", Y_2)
            np.savetxt("data/ls/T_1.txt", Y_1)
            np.savetxt("data/ls/T_2.txt", Y_2)
        if lissave:
            np.savetxt("data/lis/X_1.txt", X_1)
            np.savetxt("data/lis/X_2.txt", X_2)
    else:
        Y_1 = np.loadtxt("data/ls/Y_1.txt")
        Y_2 = np.loadtxt("data/ls/Y_2.txt")
        X_1 = np.loadtxt("data/lis/X_1.txt")
        X_2 = np.loadtxt("data/lis/X_2.txt")
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
    #np.random.default_rng().shuffle(zs, axis=1)
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


def painter(title: str, x_1, x_2, name_ys: dict, isDiffBayes: bool = False):
    plt.title(title)
    plt.plot(x_1[0], x_1[1], c="blue", marker='.', linestyle='none')
    plt.plot(x_2[0], x_2[1], c="orange", marker='.', linestyle='none')
    colors = ["green", "red", "magenta", "pink", "black", "yellow", "cyan"]
    i = 0
    for name in name_ys:
        for j in range(len(name_ys[name]["xs"])):
            if j == 0:
                plt.plot(name_ys[name]["xs"][j], name_ys[name]["ys"][j],
                         c=colors[i], label=name)
                continue
            plt.plot(name_ys[name]["xs"][j], name_ys[name]["ys"][j],
                     c=colors[i])
        if ('support_vectors' in name_ys[name]) and (name_ys[name]['support_vectors'] is not None):
            plt.plot(name_ys[name]['support_vectors']["class_0"][0, :],
                     name_ys[name]['support_vectors']["class_0"][1, :],
                     c="blue", marker='X', linestyle='none')
            plt.plot(name_ys[name]['support_vectors']["class_1"][0, :],
                     name_ys[name]['support_vectors']["class_1"][1, :],
                     c="orange", marker='X', linestyle='none')
        i += 1
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.xlim([-4, 4])
    plt.legend()
    plt.show()


def get_separating_hyperplane_top(xs: np.ndarray, w: np.ndarray):
    return xs - 1 / w[0, 0]


def get_separating_hyperplane_bottom(xs: np.ndarray, w: np.ndarray):
    return xs + 1 / w[0, 0]


if __name__ == "__main__":
    res = dict()
    config = {
        "generate": False,
        "lssave": False,
        "lissave": False,
        "checkpoints": [1]
    }
    # ==================== Синтез ЛР и ЛНР выборок двух классов ==================== #
    # 1. Синтезировать линейно разделимые выборки для двух классов двумерных случайных векторов в количестве N=100 в каждом классе
    S_1, S_2, IS_1, IS_2 = get_vectors(
        config["generate"], config["lssave"], config["lissave"])

    # ==================== SVM для ЛР классов ==================== #
    # 2. Построить линейный классификатор по методу опорных векторов на выборке с линейно разделимыми классами.
    # Использовать:
    # - задачу (7) и метод решения квадратичных задач
    # - метод sklearn.svm.SVC библиотеки scikit-learn
    # сопоставить решения из п.(б) с решением методом sklearn.svm.LinearSVC
    if 1 in config["checkpoints"]:
        training_sample_ls = generate_training_sample(S_1, S_2)

        lssvm = LSSVM(training_sample_ls)
        ys = calc_decisive_boundaries(lssvm.w, lssvm.w_n, xs)
        w, w_n = lssvm.w, lssvm.w_n
        res["LS"] = {
            "lssvm": {
                "support_vectors": {
                    "class_0": lssvm.support_vectors[0:2, lssvm.support_vectors[3, :] == -1],
                    "class_1": lssvm.support_vectors[0:2, lssvm.support_vectors[3, :] == 1]
                },
                "xs": [xs, get_separating_hyperplane_top(xs, w), get_separating_hyperplane_bottom(xs, w)],
                "ys": [ys, ys, ys],
                "p0": calc_alpha(classify_vectors(lssvm.w, lssvm.w_n, S_1, 0, 1), 1),
                "p1": calc_alpha(classify_vectors(lssvm.w, lssvm.w_n, S_2, 1, 0), 0)
            }
        }

        sklearn_svm = svm.SVC(kernel='linear')
        sklearn_svm.fit(training_sample_ls[0:3, :].T, training_sample_ls[3, :])
        w = sklearn_svm.coef_.T
        w_n = sklearn_svm.intercept_[0]
        support_vectors = training_sample_ls[:, sklearn_svm.support_]
        ys = calc_decisive_boundaries(w, w_n, xs)
        res["LS"].update({
            "sklearn_svm": {
                "support_vectors": {
                    "class_0": support_vectors[0:2, support_vectors[3, :] == -1],
                    "class_1": support_vectors[0:2, support_vectors[3, :] == 1]
                },
                "xs": [xs, get_separating_hyperplane_top(xs, w), get_separating_hyperplane_bottom(xs, w)],
                "ys": [ys, ys, ys],
                "p0": calc_alpha(classify_vectors(w, w_n, S_1, 0, 1), 1),
                "p1": calc_alpha(classify_vectors(w, w_n, S_2, 1, 0), 0)
            }
        })

        sklearn_linearSVC = svm.LinearSVC()
        sklearn_linearSVC.fit(
            training_sample_ls[0:3, :].T, training_sample_ls[3, :])
        w = sklearn_linearSVC.coef_.T
        w_n = sklearn_linearSVC.intercept_[0]
        ys = calc_decisive_boundaries(w, w_n, xs)
        res["LS"].update({
            "sklearn_linearSVC": {
                "support_vectors": None,
                "xs": [xs],
                "ys": [ys],
                "p0": calc_alpha(classify_vectors(w, w_n, S_1, 0, 1), 1),
                "p1": calc_alpha(classify_vectors(w, w_n, S_2, 1, 0), 0)
            }
        })

        painter(
            "Линейно разделимые классы", S_1, S_2, {"lssvm": res["LS"]["lssvm"]}
        )
        # painter(
        #     "Линейно разделимые классы", S_1, S_2, {}
        # )
        # painter(
        #     "Линейно разделимые классы", S_1, S_2, res["LS"]
        # )

    # ==================== SVM для ЛНР классов ==================== #
    # 3. Построить линейный классификатор по SVM на выборке с линейно неразделимыми классами.
    # Использовать:
    # - задачу (12) и метод решения квадратичных задач.
    #       Указать решения для C=1/10, 1, 10 и подобранно самостоятельно «лучшим коэффициентом».
    # - метод sklearn.svm.SVC библиотеки scikit-learn
    if 2 in config["checkpoints"]:
        training_sample_ls = generate_training_sample(IS_1, IS_2)
        Cs = [10]

        for C in Cs:
            lissvm = LISSVM(training_sample_ls, C)
            ys = calc_decisive_boundaries(lissvm.w, lissvm.w_n, xs)
            res[f"LIS_{C}"] = {
                "lissvm": {
                    "support_vectors": {
                        "class_0": lissvm.support_vectors[0:2, lissvm.support_vectors[3, :] == -1],
                        "class_1": lissvm.support_vectors[0:2, lissvm.support_vectors[3, :] == 1]
                    },
                    "xs": [xs, xs, xs],
                    "ys": [ys, get_separating_hyperplane_top(ys, lissvm.w, lissvm.w_n), get_separating_hyperplane_bottom(ys, lissvm.w, lissvm.w_n)],
                    "p0": calc_alpha(classify_vectors(lissvm.w, lissvm.w_n, S_1, 0, 1), 1),
                    "p1": calc_alpha(classify_vectors(lissvm.w, lissvm.w_n, S_2, 1, 0), 0)
                }
            }
            # print("Support vectors count: ", lissvm.support_vectors.shape)
            # print("Support vectors count class 0: ",
            #       lissvm.support_vectors[0:2, lissvm.support_vectors[3, :] == -1].shape[1])
            # print("Support vectors count class 1: ",
            #       lissvm.support_vectors[0:2, lissvm.support_vectors[3, :] == 1].shape[1])
            # print("Support vectors indexes: ")
            # stry = ''
            # for i in range(lissvm.support_vectors_indexes.shape[0]):
            #     if lissvm.support_vectors_indexes[i]:
            #         stry += f" {i};"
            # print("\t", stry)
            # print("Support vectors: \n")
            # for i in range(lissvm.support_vectors.shape[1]):
            #     print(
            #         i, ": ", lissvm.support_vectors[0, i], lissvm.support_vectors[1, i])
            # print("Training sample: \n")
            # for i in range(training_sample_ls.shape[1]):
            #     if lissvm.lyambdas[i] >= 0.01:
            #         print(
            #             i, ": ", training_sample_ls[0, i], training_sample_ls[1, i], "lyambda: ", lissvm.lyambdas[i])

            # print("Training sample: \n")
            # for i in range(training_sample_ls.shape[1]):
            #     print(
            #         i, ": ", training_sample_ls[0, i], training_sample_ls[1, i], "lyambda: ", lissvm.lyambdas[i])
            # print(np.unique(IS_1).shape)
            # print(np.unique(IS_2).shape)

            sklearn_svm = svm.SVC(kernel='linear', C=C)
            sklearn_svm.fit(
                training_sample_ls[0:3, :].T, training_sample_ls[3, :])
            w = sklearn_svm.coef_.T
            w_n = sklearn_svm.intercept_[0]
            support_vectors = training_sample_ls[:, sklearn_svm.support_]
            ys = calc_decisive_boundaries(w, w_n, xs)
            res[f"LIS_{C}"].update({
                "sklearn_svm": {
                    "support_vectors": {
                        "class_0": support_vectors[0:2, support_vectors[3, :] == -1],
                        "class_1": support_vectors[0:2, support_vectors[3, :] == 1]
                    },
                    "xs": [xs, get_separating_hyperplane_top(xs, w), get_separating_hyperplane_bottom(xs, w)],
                    "ys": [ys, ys, ys],
                    "p0": calc_alpha(classify_vectors(w, w_n, S_1, 0, 1), 1),
                    "p1": calc_alpha(classify_vectors(w, w_n, S_2, 1, 0), 0)
                }
            })

            painter(
                f"Линейно разделимые классы. С = {C}", IS_1, IS_2, res[f"LIS_{C}"]
            )
            print()

    # 4. Построить классификатор по SVM, разделяющий линейно неразделимые классы.
    # Использовать:
    # - задачу (14) и метод решения квадратичных задач,
    #       Исследовать решение для различных значений параметра C=1/10, 1, 10 и различных ядер из таблицы 1
    # - метод sklearn.svm.SVC.
    if 3 in config["checkpoints"]:
        training_sample_ls = generate_training_sample(IS_1, IS_2)
        Cs = [0.1, 1, 10]
        params = {"d": 3, "c": 1}

        for C in Cs:
            klissvm = KLISSVM(training_sample_ls, C,
                              kernel="polynomial", params=params)
            ys = calc_decisive_boundaries(klissvm.w, klissvm.w_n, xs)
            res[f"KLIS_{C}"] = {
                "klissvm": {
                    "support_vectors": {
                        "class_0": klissvm.support_vectors[0:2, klissvm.support_vectors[3, :] == -1],
                        "class_1": klissvm.support_vectors[0:2, klissvm.support_vectors[3, :] == 1]
                    },
                    "xs": [xs, xs, xs],
                    "ys": [ys, get_separating_hyperplane_top(ys, klissvm.w), get_separating_hyperplane_bottom(ys, klissvm.w)],
                    "p0": calc_alpha(classify_vectors(klissvm.w, klissvm.w_n, S_1, 0, 1), 1),
                    "p1": calc_alpha(classify_vectors(klissvm.w, klissvm.w_n, S_2, 1, 0), 0)
                }
            }

        y = np.linspace(-5, 5, N * 2)
        x = np.linspace(-5, 5, N * 2)
        # create cooordinate grid
        xx, yy = np.meshgrid(x, y)
        xy = np.vstack((xx.ravel(), yy.ravel())).T

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
