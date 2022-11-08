import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from vector_generator import generate_norm_vector
from fisher import Fisher
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


if __name__ == "__main__":
    N = 200
    M_1, M_2 = get_Ms()
    B_1, B_2 = get_Bs()
    Bs_e = np.array([B_1, B_1])
    Bs = np.array([B_1, B_2])
    Ms = np.array([M_1, M_2])
    Ps = np.array([0.5, 0.5])
    Y_1 = generate_norm_vector(N, B_1, M_1)
    Y_2 = generate_norm_vector(N, B_1, M_2)
    X_1 = generate_norm_vector(N, B_1, M_1)
    X_2 = generate_norm_vector(N, B_2, M_2)

    active_points = [1]
    if 1 in active_points:
        # Равные корреляционные матрицы
        xs = np.linspace(-3, 3, N)
        fisher = Fisher(M_1, M_2, B_1, B_1)
        ys_f = fisher.calc_decisive_boundaries(xs)
        ys_b = d_lj_equal_B(xs, B_1, M_1, M_2, 0.5, 0.5)
        plt.scatter(Y_1[0], Y_1[1])
        plt.scatter(Y_2[0], Y_2[1])
        plt.plot(xs, ys_f)
        plt.plot(xs, ys_b)
        plt.xlim([-4, 4])
        plt.show()

        cbr_0 = classify_vectors(Y_1, Bs_e, Ms, Ps)
        cbr_1 = classify_vectors(Y_2, Bs_e, Ms, Ps)
        cfr_0 = fisher.classify_vectors(Y_1, 0, 1)
        cfr_1 = fisher.classify_vectors(Y_2, 1, 0)
        print(cbr_0, "\n", cbr_1)
        print(cfr_0, "\n", cfr_1)
        print("Равные корреляционные матрицы: ")
        print("\tБК: Экспериментальные ошибки классификации для классов 0 и 1: ",
              calc_allha(cbr_0, 1), " , ", calc_allha(cbr_1, 0))
        print("\tКФ: Экспериментальные ошибки классификации для классов 0 и 1: ",
              calc_allha(cfr_0, 1), " , ", calc_allha(cfr_1, 0))

        # Разные корреляционные матрицы
        xs = np.linspace(-3, 3, N)
        fisher = Fisher(M_1, M_2, B_1, B_2)
        ys_f = fisher.calc_decisive_boundaries(xs)
        ys_b = d_lj_different_B(xs, B_1, B_2, M_1, M_2)
        plt.scatter(X_1[0], X_1[1])
        plt.scatter(X_2[0], X_2[1])
        plt.plot(xs, ys_f)
        plt.plot(ys_b[0], xs)
        plt.plot(ys_b[1], xs)
        plt.xlim([-4, 4])
        plt.show()

        cbr_0 = classify_vectors(X_1, Bs, Ms, Ps)
        cbr_1 = classify_vectors(X_2, Bs, Ms, Ps)
        cfr_0 = fisher.classify_vectors(X_1, 0, 1)
        cfr_1 = fisher.classify_vectors(X_2, 1, 0)
        print("Разные корреляционные матрицы: ")
        print("\tБК: Экспериментальные ошибки классификации для классов 0 и 1: ",
              calc_allha(cbr_0, 1), " , ", calc_allha(cbr_1, 0))
        print("\tКФ: Экспериментальные ошибки классификации для классов 0 и 1: ",
              calc_allha(cfr_0, 1), " , ", calc_allha(cfr_1, 0))
