import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from vector_generator import generate_norm_vector
from fisher import Fisher
from stdmin import STDMin
from bayes import *


def get_Ms():
    M_1 = np.array([
        [-1],
        [1]
    ])
    M_2 = np.array([
        [0],
        [-1]
    ])
    return M_1, M_2


def get_Bs():
    B_1 = np.array([
        [0.4, 0.3],
        [0.3, 0.5]
    ])
    B_2 = np.array([
        [0.3, 0],
        [0, 0.3]
    ])
    return B_1, B_2


def calc_allha(classify_sequence: np.ndarray, class_type: int):
    return len(classify_sequence[classify_sequence == class_type]) / len(classify_sequence)


def calc_bayes(xs, X_1, X_2, M_i, M_j, B_i, B_j):
    ys_b = None
    if np.array_equal(B_i, B_j):
        ys_b = d_lj_equal_B(xs, B_i, M_i, M_j, 0.5, 0.5)
    else:
        ys_b = d_lj_different_B(xs, B_i, B_j, M_i, M_j)
    Bs, Ms = np.array([B_i, B_j]), np.array([M_i, M_j])
    Ps = np.array([0.5, 0.5])
    cbr_0 = classify_vectors(X_1, Bs, Ms, Ps)
    cbr_1 = classify_vectors(X_2, Bs, Ms, Ps)
    return ys_b, cbr_0, cbr_1


def calc_fisher(xs, X_1, X_2, M_i, M_j, B_i, B_j):
    fisher = Fisher(M_i, M_j, B_i, B_j)
    ys_f = fisher.calc_decisive_boundaries(xs)
    cfr_0 = fisher.classify_vectors(X_1, 0, 1)
    cfr_1 = fisher.classify_vectors(X_2, 1, 0)
    return ys_f, cfr_0, cfr_1


def calc_stdmin(xs, X_1, X_2):
    m = 400
    stdmin = STDMin(m, X_1, X_2)
    ys_std = stdmin.calc_decisive_boundaries(xs)
    cstdr_0 = stdmin.classify_vectors(X_1, 0, 1)
    cstdr_1 = stdmin.classify_vectors(X_2, 1, 0)
    return ys_std, cstdr_0, cstdr_1


def demonstration_2_plot(title, xs, X_1, X_2, ys_db_1, ys_db_2, ys_db_1_name, ys_db_2_name, isDiffBayes=False):
    plt.title(title)
    plt.scatter(X_1[0], X_1[1])
    plt.scatter(X_2[0], X_2[1])
    if isDiffBayes:
        plt.plot(ys_db_1, xs, label=ys_db_1_name, c="red")
    else:
        plt.plot(xs, ys_db_1, label=ys_db_1_name, c="red")
    plt.plot(xs, ys_db_2, label=ys_db_2_name, c="green")
    plt.xlim([-4, 4])
    plt.legend()
    plt.show()


def demonstration_errors(title, classifier_errors):
    print(title)
    for classifier in classifier_errors:
        print(f"\t{classifier}: Экспериментальные ошибки классификации для классов 0 и 1: ",
              classifier_errors[classifier][0], ", ", classifier_errors[classifier][1])


if __name__ == "__main__":
    N = 200
    M_1, M_2 = get_Ms()
    B_1, B_2 = get_Bs()
    Bs_e = np.array([B_1, B_1])
    Bs = np.array([B_1, B_2])
    Ms = np.array([M_1, M_2])
    Ps = np.array([0.5, 0.5])
    xs = np.linspace(-3, 3, N)
    Y_1 = generate_norm_vector(N, B_1, M_1)
    Y_2 = generate_norm_vector(N, B_1, M_2)
    X_1 = generate_norm_vector(N, B_1, M_1)
    X_2 = generate_norm_vector(N, B_2, M_2)

    exp_res = dict()
    ys_b, cbr_0, cbr_1 = calc_bayes(xs, Y_1, Y_2, M_1, M_2, B_1, B_1)
    ys_b_d, cbr_0_d, cbr_1_d = calc_bayes(xs, X_1, X_2, M_1, M_2, B_1, B_2)
    exp_res["bayes"] = {
        "B_equal": {"ys": ys_b, "p0": calc_allha(cbr_0, 1), "p1": calc_allha(cbr_1, 0)},
        "B_diff": {"ys": ys_b_d, "p0": calc_allha(cbr_0_d, 1), "p1": calc_allha(cbr_1_d, 0)}
    }
    ys_f, cfr_0, cfr_1 = calc_fisher(xs, Y_1, Y_2, M_1, M_2, B_1, B_1)
    ys_f_d, cfr_0_d, cfr_1_d = calc_fisher(xs, X_1, X_2, M_1, M_2, B_1, B_2)
    exp_res["fisher"] = {
        "B_equal": {"ys": ys_f, "p0": calc_allha(cfr_0, 1), "p1": calc_allha(cfr_1, 0)},
        "B_diff": {"ys": ys_f_d, "p0": calc_allha(cfr_0_d, 1), "p1": calc_allha(cfr_1_d, 0)}
    }
    ys_std, cstdr_0, cstdr_1 = calc_stdmin(xs, Y_1, Y_2)
    ys_std_d, cstdr_0_d, cstdr_1_d = calc_stdmin(xs, X_1, X_2)
    exp_res["stdmin"] = {
        "B_equal": {"ys": ys_std, "p0": calc_allha(cstdr_0, 1), "p1": calc_allha(cstdr_1, 0)},
        "B_diff": {"ys": ys_std_d, "p0": calc_allha(cstdr_0_d, 1), "p1": calc_allha(cstdr_1_d, 0)}
    }

    active_points = [2]
    if 1 in active_points:
        # Равные корреляционные матрицы
        title = "Равные корреляционные матрицы"
        demonstration_2_plot(
            title, xs, Y_1, Y_2, exp_res["bayes"]["B_equal"]["ys"], exp_res["fisher"]["B_equal"]["ys"], "Байес", "Фишер")
        demonstration_errors(title, {
            "БК": (exp_res["bayes"]["B_equal"]["p0"], exp_res["bayes"]["B_equal"]["p0"]),
            "КФ": (exp_res["fisher"]["B_equal"]["p0"], exp_res["fisher"]["B_equal"]["p0"])})

        # Разные корреляционные матрицы
        title = "Разные корреляционные матрицы: "
        demonstration_2_plot(title, xs, X_1, X_2, exp_res["bayes"]["B_diff"]["ys"]
                             [0], exp_res["fisher"]["B_diff"]["ys"], "Байес", "Фишер", isDiffBayes=True)
        demonstration_errors(title, {
            "БК": (exp_res["bayes"]["B_diff"]["p0"], exp_res["bayes"]["B_diff"]["p1"]),
            "КФ": (exp_res["fisher"]["B_diff"]["p0"], exp_res["fisher"]["B_diff"]["p1"])})

    if 2 in active_points:
        # Равные корреляционные матрицы
        title = "Равные корреляционные матрицы"
        demonstration_2_plot(
            title, xs, Y_1, Y_2, exp_res["bayes"]["B_equal"]["ys"], exp_res["stdmin"]["B_equal"]["ys"], "Байес", "СКО")
        demonstration_errors(title, {
            "БК": (exp_res["bayes"]["B_equal"]["p0"], exp_res["bayes"]["B_equal"]["p0"]),
            "СКО": (exp_res["stdmin"]["B_equal"]["p0"], exp_res["stdmin"]["B_equal"]["p0"])})

        # Разные корреляционные матрицы
        title = "Разные корреляционные матрицы: "
        demonstration_2_plot(title, xs, X_1, X_2, exp_res["bayes"]["B_diff"]["ys"]
                             [0], exp_res["stdmin"]["B_diff"]["ys"], "Байес", "СКО", isDiffBayes=True)
        demonstration_errors(title, {
            "БК": (exp_res["bayes"]["B_diff"]["p0"], exp_res["bayes"]["B_diff"]["p1"]),
            "СКО": (exp_res["stdmin"]["B_diff"]["p0"], exp_res["stdmin"]["B_diff"]["p1"])})
