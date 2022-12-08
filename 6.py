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
kernel_dict = {
    "poly": "Полиномиальное",
    "sigmoid": "Сигмоидальная функция",
    "rbf": "Радиальная функция",
    "rbf_gauss": "Радиальная функция Гаусса",
}


def get_vectors(generate: bool, lssave: bool, lissave: bool):
    if generate:
        Y_1 = generate_norm_vector(N, B_1, M_1_ls)
        Y_2 = generate_norm_vector(N, B_2, M_2_ls)
        X_1 = generate_norm_vector(N, B_1, M_1_lis)
        X_2 = generate_norm_vector(N, B_2, M_2_lis)
        if lssave:
            np.savetxt("data/ls/Y_1.txt", Y_1)
            np.savetxt("data/ls/Y_2.txt", Y_2)
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
    zs = np.ndarray(shape=(3, m))
    for i in range(m):
        if i % 2 == 0:
            zs[0, i] = x_1[0, int((i / 2) % length)]
            zs[1, i] = x_1[1, int((i / 2) % length)]
            zs[2, i] = - 1
        else:
            zs[0, i] = x_2[0, int((i - 1 / 2) % length)]
            zs[1, i] = x_2[1, int((i - 1 / 2) % length)]
            zs[2, i] = 1
    # Дополнительное переупорядочивание
    # np.random.default_rng().shuffle(zs, axis=1)
    return zs


def generate_training_sample_2(x_1: np.ndarray, x_2: np.ndarray):
    m = x_1.shape[1] + x_2.shape[1]
    zs = np.ndarray(shape=(3, m))
    j = 0
    for i in range(m // 2):
        zs[0, j] = x_1[0, i]
        zs[1, j] = x_1[1, i]
        zs[2, j] = - 1
        j += 1

        zs[0, j] = x_2[0, i]
        zs[1, j] = x_2[1, i]
        zs[2, j] = 1
        j += 1
    return zs


def calc_alpha(classify_sequence: np.ndarray, class_type: int):
    return len(classify_sequence[classify_sequence == class_type]) / len(classify_sequence)


def calc_decisive_boundary(w: np.ndarray, w_n: float, x: float):
    return (- w[0, 0] * x - w_n) / w[1, 0]


def calc_decisive_boundaries(w: np.ndarray, w_n: float, xs: np.ndarray):
    v_calc_bound = np.vectorize(calc_decisive_boundary, excluded=[0])
    return v_calc_bound(w, w_n, xs)


def calc_decisive_function(w: np.ndarray, w_n: float, x: np.ndarray):
    return w[0, 0] * x[0] + w[1, 0] * x[1] + w_n


def calc_decisive_function_solveqp(x: np.ndarray, kernel_func, w_n):
    return kernel_func(x) + w_n


def calc_decisive_functions(w: np.ndarray, w_n: float, xs: np.ndarray):
    v_calc_dec_func = np.vectorize(calc_decisive_function,
                                   excluded=[0], signature='(),(n)->()')
    return v_calc_dec_func(w, w_n, np.reshape(xs.T, newshape=(xs.shape[1], xs.shape[0])))


def calc_decisive_functions_solveqp(xs: np.ndarray, kernel_func, w_n):
    return np.array([calc_decisive_function_solveqp(xs[:, i], kernel_func, w_n) for i in range(xs.shape[1])])


def classify_vectors(w: np.ndarray, w_n: float, xs: np.ndarray, class_type: int, another_class_type: int):
    ds = calc_decisive_functions(w, w_n, xs)
    if class_type == 0:
        return np.where(ds < 0, class_type, another_class_type)
    return np.where(ds > 0, class_type, another_class_type)


def classify_vectors_solveqp(xs: np.ndarray, kernel_func, w_n: float, class_type: int, another_class_type: int):
    ds = calc_decisive_functions_solveqp(xs, kernel_func, w_n)
    if class_type == 0:
        return np.where(ds < 0, class_type, another_class_type)
    return np.where(ds > 0, class_type, another_class_type)


def classify_vectors_svc(xs: np.ndarray, class_type: int, another_class_type: int):
    ds = sklearn_svm.predict(xs.T)
    if class_type == 0:
        return np.where(ds < 0, class_type, another_class_type)
    return np.where(ds > 0, class_type, another_class_type)


def painter(
        title: str, x_1, x_2, name_ys: dict,
        isDiffBayes: bool = False, printErrors: bool = False,
        delay_show: bool = False, separate_plot=True):
    if separate_plot:
        plt.figure()
    plt.title(title)
    plt.plot(x_1[0], x_1[1], c="blue", marker='.', linestyle='none')
    plt.plot(x_2[0], x_2[1], c="orange", marker='.', linestyle='none')
    colors = ["green", "red", "magenta", "pink", "black", "yellow", "cyan"]
    i = 0
    proxy, legends = [], []
    for name in name_ys:
        if ('yy' in name_ys[name]):
            cs = plt.contour(name_ys[name]["xx"], name_ys[name]["yy"], name_ys[name]["ds"],
                             levels=[-1, 0, 1], colors=[colors[i], colors[i], colors[i]])
            h1, _ = cs.legend_elements()
            proxy.append(h1[0])
            legends.append(name)
        else:
            for j in range(len(name_ys[name]["xs"])):
                if j == 0:
                    plt.plot(name_ys[name]["xs"][j], name_ys[name]["ys"][j],
                             c=colors[i], label=name)
                    continue
                plt.plot(name_ys[name]["xs"][j], name_ys[name]["ys"][j],
                         c=colors[i])
        if ('support_vectors' in name_ys[name]) and (name_ys[name]['support_vectors'] is not None):
            marker = 'X'
            plt.plot(name_ys[name]['support_vectors']["class_0"][0, :],
                     name_ys[name]['support_vectors']["class_0"][1, :],
                     c="blue", marker=marker, linestyle='none')
            plt.plot(name_ys[name]['support_vectors']["class_1"][0, :],
                     name_ys[name]['support_vectors']["class_1"][1, :],
                     c="orange", marker=marker, linestyle='none')
        i += 1
        if printErrors:
            p_0, p_1 = name_ys[name]["p0"], name_ys[name]["p1"]
            print(
                name, f"\n\tВероятности ошибочной классификации: \tp_0: {p_0}\tp_1: {p_1}")

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    scale = 1.5
    xlim_min, xlim_max = np.min([x_1[0], x_2[0]]) - \
        scale, np.max([x_1[0], x_2[0]]) + scale
    ylim_min, ylim_max = np.min([x_1[1], x_2[1]]) - \
        scale, np.max([x_1[1], x_2[1]]) + scale
    plt.xlim([xlim_min, xlim_max])
    plt.ylim([ylim_min, ylim_max])
    if len(proxy) == 0:
        plt.legend()
    else:
        plt.legend(proxy, legends)
    if delay_show is False:
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
        "checkpoints": [1, 2, 3]
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
        # training_sample_ls = generate_training_sample(S_1, S_2)
        training_sample_ls = generate_training_sample_2(S_1, S_2)
        # print("Fisrt", np.unique(training_sample_ls).shape)
        # print("Second", np.unique(training_sample_ls_2).shape)

        lssvm = LSSVM(training_sample_ls)
        ys = calc_decisive_boundaries(lssvm.w, lssvm.w_n, xs)
        w, w_n = lssvm.w, lssvm.w_n
        res["LS"] = {
            "lssvm": {
                "support_vectors": {
                    "class_0": lssvm.support_vectors[0:2, lssvm.support_vectors[2, :] == -1],
                    "class_1": lssvm.support_vectors[0:2, lssvm.support_vectors[2, :] == 1]
                },
                "xs": [xs, get_separating_hyperplane_top(xs, w), get_separating_hyperplane_bottom(xs, w)],
                "ys": [ys, ys, ys],
                "p0": calc_alpha(classify_vectors(lssvm.w, lssvm.w_n, S_1, 0, 1), 1),
                "p1": calc_alpha(classify_vectors(lssvm.w, lssvm.w_n, S_2, 1, 0), 0)
            }
        }

        sklearn_svm = svm.SVC(kernel='linear')
        sklearn_svm.fit(training_sample_ls[0:2, :].T, training_sample_ls[2, :])
        w = sklearn_svm.coef_.T
        w_n = sklearn_svm.intercept_[0]
        support_vectors = training_sample_ls[:, sklearn_svm.support_]
        ys = calc_decisive_boundaries(w, w_n, xs)
        res["LS"].update({
            "sklearn_svm": {
                "support_vectors": {
                    "class_0": support_vectors[0:2, support_vectors[2, :] == -1],
                    "class_1": support_vectors[0:2, support_vectors[2, :] == 1]
                },
                "xs": [xs, get_separating_hyperplane_top(xs, w), get_separating_hyperplane_bottom(xs, w)],
                "ys": [ys, ys, ys],
                "p0": calc_alpha(classify_vectors(w, w_n, S_1, 0, 1), 1),
                "p1": calc_alpha(classify_vectors(w, w_n, S_2, 1, 0), 0)
            }
        })

        sklearn_linearSVC = svm.LinearSVC()
        sklearn_linearSVC.fit(
            training_sample_ls[0:2, :].T, training_sample_ls[2, :])
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

        # painter(
        #     "Линейно разделимые классы", S_1, S_2, {"lssvm": res["LS"]["lssvm"]}, printErrors=True
        # )
        # painter(
        #     "Линейно разделимые классы", S_1, S_2, {"sklearn_svm": res["LS"]["sklearn_svm"]}, printErrors=True
        # )
        painter(
            "Линейно разделимые классы", S_1, S_2, res["LS"], printErrors=True
        )

    # ==================== SVM для ЛНР классов ==================== #
    # 3. Построить линейный классификатор по SVM на выборке с линейно неразделимыми классами.
    # Использовать:
    # - задачу (12) и метод решения квадратичных задач.
    #       Указать решения для C=1/10, 1, 10 и подобранно самостоятельно «лучшим коэффициентом».
    # - метод sklearn.svm.SVC библиотеки scikit-learn
    if 2 in config["checkpoints"]:
        training_sample_lis = generate_training_sample_2(IS_1, IS_2)
        Cs = [0.1, 1, 10, 18]

        for C in Cs:
            lissvm = LISSVM(training_sample_lis, C)
            w, w_n = lissvm.w, lissvm.w_n
            ys = calc_decisive_boundaries(w, w_n, xs)
            res[f"LIS_{C}"] = {
                "lissvm": {
                    "support_vectors": {
                        "class_0": lissvm.support_vectors[0:2, lissvm.support_vectors[2, :] == -1],
                        "class_1": lissvm.support_vectors[0:2, lissvm.support_vectors[2, :] == 1]
                    },
                    "xs": [xs, get_separating_hyperplane_top(xs, w), get_separating_hyperplane_bottom(xs, w)],
                    "ys": [ys, ys, ys],
                    "p0": calc_alpha(classify_vectors(w, w_n, IS_1, 0, 1), 1),
                    "p1": calc_alpha(classify_vectors(w, w_n, IS_2, 1, 0), 0)
                }
            }

            sklearn_svm = svm.SVC(kernel='linear', C=C)
            sklearn_svm.fit(
                training_sample_lis[0:2, :].T, training_sample_lis[2, :])
            w = sklearn_svm.coef_.T
            w_n = sklearn_svm.intercept_[0]
            support_vectors = training_sample_lis[:, sklearn_svm.support_]
            ys = calc_decisive_boundaries(w, w_n, xs)
            res[f"LIS_{C}"].update({
                "sklearn_svm": {
                    "support_vectors": {
                        "class_0": support_vectors[0:2, support_vectors[2, :] == -1],
                        "class_1": support_vectors[0:2, support_vectors[2, :] == 1]
                    },
                    "xs": [xs, get_separating_hyperplane_top(xs, w), get_separating_hyperplane_bottom(xs, w)],
                    "ys": [ys, ys, ys],
                    "p0": calc_alpha(classify_vectors(w, w_n, IS_1, 0, 1), 1),
                    "p1": calc_alpha(classify_vectors(w, w_n, IS_2, 1, 0), 0)
                }
            })

            # painter(
            #     f"Линейно неразделимые классы. С = {C}", IS_1, IS_2, {"lissvm": res[f"LIS_{C}"]["lissvm"]}, printErrors=True, delay_show=True
            # )
            painter(
                f"Линейно неразделимые классы. С = {C}", IS_1, IS_2, res[f"LIS_{C}"], printErrors=True, delay_show=True
            )
            plt.show()

    # ==================== SVM c ЯДРОМ для ЛНР классов ==================== #
    # 4. Построить классификатор по SVM, разделяющий линейно неразделимые классы.
    # Использовать:
    # - задачу (14) и метод решения квадратичных задач,
    #       Исследовать решение для различных значений параметра C=1/10, 1, 10 и различных ядер из таблицы 1
    # - метод sklearn.svm.SVC.
    if 3 in config["checkpoints"]:
        training_sample_lis = generate_training_sample_2(IS_1, IS_2)
        x_1, x_2 = IS_1, IS_2
        scale = 1.5
        xlim_min, xlim_max = np.min([x_1[0], x_2[0]]) - \
            scale, np.max([x_1[0], x_2[0]]) + scale
        ylim_min, ylim_max = np.min([x_1[1], x_2[1]]) - \
            scale, np.max([x_1[1], x_2[1]]) + scale
        y = np.linspace(ylim_min, ylim_max, N * 2)
        x = np.linspace(xlim_min, xlim_max, N * 2)
        xx, yy = np.meshgrid(x, y)
        xy = np.vstack((xx.ravel(), yy.ravel())).T

        Cs = [10]
        # ['poly', 'sigmoid', 'rbf', 'rbf_gauss']
        kernels = ['poly', 'sigmoid', 'rbf', 'rbf_gauss']
        params = {
            "d": 3, "c_p": 1,
            "g_s": 0.08, "c_s": -0.6,
            "g_r": 1, "g_r_gauss": 1 / (2 * np.var(np.sqrt(training_sample_lis ** 2 + training_sample_lis ** 2)))
        }
        lonely_mode = True

        for kernel in kernels:
            for C in Cs:
                klissvm = KLISSVM(training_sample_lis, C,
                                  kernel=kernel, params=params)
                res[f"KLIS_{kernel}_{C}"] = {
                    "klissvm": {
                        "support_vectors": {
                            "class_0": klissvm.support_vectors[0:2, klissvm.support_vectors[2, :] == -1],
                            "class_1": klissvm.support_vectors[0:2, klissvm.support_vectors[2, :] == 1]
                        },
                        "xx": xx,
                        "yy": yy,
                        "ds": np.array([klissvm.get_discriminant_kernel(xy[i, :].T) + klissvm.w_n for i in range(xy.shape[0])]).reshape(xx.shape),
                        "p0": calc_alpha(classify_vectors_solveqp(IS_1, klissvm.get_discriminant_kernel, klissvm.w_n, 0, 1), 1),
                        "p1": calc_alpha(classify_vectors_solveqp(IS_2, klissvm.get_discriminant_kernel, klissvm.w_n, 1, 0), 0)
                    }
                }

                sklearn_svm = None
                if kernel == 'poly':
                    sklearn_svm = svm.SVC(kernel=kernel, C=C,
                                          degree=params["d"], coef0=params["c_p"])
                if kernel == 'sigmoid':
                    sklearn_svm = svm.SVC(
                        kernel=kernel, C=C, coef0=params["c_s"], gamma=params["g_s"])
                if kernel == 'rbf':
                    sklearn_svm = svm.SVC(
                        kernel=kernel, C=C, gamma=params["g_r"])
                if kernel == 'rbf_gauss':
                    sklearn_svm = svm.SVC(
                        kernel='rbf', C=C, gamma=params["g_r_gauss"])
                sklearn_svm.fit(
                    training_sample_lis[0:2, :].T, training_sample_lis[2, :])
                w_n = sklearn_svm.intercept_[0]
                support_vectors = training_sample_lis[:,
                                                      sklearn_svm.support_]
                res[f"KLIS_{kernel}_{C}"].update({
                    "sklearn_svm": {
                        "support_vectors": {
                            "class_0": support_vectors[0:2, support_vectors[2, :] == -1],
                            "class_1": support_vectors[0:2, support_vectors[2, :] == 1]
                        },
                        "xx": xx,
                        "yy": yy,
                        "ds": sklearn_svm.decision_function(xy).reshape(xx.shape),
                        "p0": calc_alpha(classify_vectors_svc(IS_1, 0, 1), 1),
                        "p1": calc_alpha(classify_vectors_svc(IS_2, 0, 1), 0)
                    }
                })

                if lonely_mode:
                    painter(
                        f"Линейно неразделимые классы. С = {C}. Ядро - {kernel_dict[kernel]}",
                        IS_1, IS_2, res[f"KLIS_{kernel}_{C}"],
                        delay_show=True,
                        separate_plot=True,
                        printErrors=True
                    )
        if lonely_mode:
            plt.show()

        if 1 == 1:
            fig = plt.figure()
            plot_num = 1
            for kernel in kernels:
                fig.add_subplot(2, 2, plot_num)
                for C in Cs:
                    painter(
                        f"С = {C}. Ядро - {kernel_dict[kernel]}",
                        IS_1, IS_2, res[f"KLIS_{kernel}_{C}"],
                        delay_show=True,
                        printErrors=True,
                        separate_plot=False
                    )
                plot_num += 1
            plt.show()
