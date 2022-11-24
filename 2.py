import math
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import modules.bayes as bayes
import utils.constants as const
from utils.vector_generator import generate_norm_vector
from scipy.special import erf, erfinv


def Phi(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def invPhi(x):
    return np.sqrt(2) * erfinv(2 * x - 1)


def get_Mahalanobis_distance(M_i, M_j, B):
    return np.dot(np.dot((M_j - M_i).T, la.inv(B)), (M_j - M_i))


def d_lj_equal_Neyman_Pearson(xs: np.ndarray, B: np.ndarray,
                              M_i: np.ndarray, M_j: np.ndarray, p_0: float) -> np.ndarray:
    l = np.dot((M_j - M_i).T, la.inv(B))
    r = (- 0.5) * np.dot(np.dot((M_j + M_i).T, la.inv(B)), (M_j - M_i))[0, 0]
    m_d = get_Mahalanobis_distance(M_i, M_j, B)
    lambda_tilda = - 0.5 * m_d + np.sqrt(m_d) * invPhi(1 - p_0)
    ys = np.copy(xs)
    for i in range(len(xs)):
        x = xs[i]
        ys[i] = (-x * l[0, 1] - r - lambda_tilda) / l[0, 0]
    return ys


def get_p_error(B: np.ndarray, M_i: np.ndarray, M_j: np.ndarray, P_i: float, P_j: float) -> (float, float):
    m_d = get_Mahalanobis_distance(M_i, M_j, B)
    lambda_tilda = P_i / P_j
    p_10 = Phi((lambda_tilda - 0.5 * m_d) / np.sqrt(m_d))[0, 0]
    p_01 = 1 - Phi((lambda_tilda + 0.5 * m_d) / np.sqrt(m_d))[0, 0]
    return p_01, p_10


def probability(X, M_i, M_j, B_i, B_j, P_i, P_j):
    count = 0
    for i in range(X.shape[1]):
        x = np.array(X[0, i], X[1, i])
        d1 = math.log(P_i) - math.log(np.sqrt(la.det(B_i))) - \
            0.5 * np.transpose(x - M_i) @ la.inv(B_i) @ (x - M_i)
        d2 = math.log(P_j) - math.log(np.sqrt(la.det(B_j))) - \
            0.5 * np.transpose(x - M_j) @ la.inv(B_j) @ (x - M_j)
        if d2 > d1:
            count += 1
    return count / X.shape[1]


if __name__ == '__main__':
    N = 200
    M_1, M_2, M_3 = const.M_1, const.M_2, const.M_3
    B_1, B_2, B_3 = const.B_1, const.B_2, const.B_3
    Y_1 = generate_norm_vector(N, B_1, M_1)
    Y_2 = generate_norm_vector(N, B_1, M_2)
    X_1 = generate_norm_vector(N, B_1, M_1)
    X_2 = generate_norm_vector(N, B_2, M_2)
    X_3 = generate_norm_vector(N, B_3, M_3)

    # MENU
    active_points = [1, 2.1, 2.2, 3]
    if 1 in active_points:
        xs = np.linspace(-3, 3, N)
        ys = bayes.d_lj_equal_B(xs, B_1, M_1, M_2, 0.5, 0.5)
        plt.scatter(Y_1[0], Y_1[1])
        plt.scatter(Y_2[0], Y_2[1])
        plt.plot(xs, ys)
        plt.xlim([-4, 4])
        plt.show()

        p_error = get_p_error(B_1, M_1, M_2, 0.5, 0.5)
        print("Вероятности ошибочной классификации: ", p_error)
        p_error_total_sum = (p_error[0] + p_error[1])
        print("Суммарная вероятность ошибочной классификации: ", p_error_total_sum)

    if 2.1 in active_points:
        xs = np.linspace(-3, 3, N)
        ys = bayes.d_lj_equal_B(xs, B_1, M_1, M_2, 0.5, 0.5)
        plt.scatter(Y_1[0], Y_1[1])
        plt.scatter(Y_2[0], Y_2[1])
        plt.plot(xs, ys)
        plt.xlim([-4, 4])
        plt.show()

    if 2.2 in active_points:
        p_0 = 0.05
        xs = np.linspace(-3, 3, N)
        ys = d_lj_equal_Neyman_Pearson(xs, B_1, M_1, M_2, p_0)
        plt.scatter(Y_1[0], Y_1[1])
        plt.scatter(Y_2[0], Y_2[1])
        plt.plot(xs, ys)
        plt.xlim([-4, 4])
        plt.show()

    if 3 in active_points:
        plt.scatter(X_1[0], X_1[1], c='b')
        plt.scatter(X_2[0], X_2[1], c='g')
        plt.scatter(X_3[0], X_3[1], c='r')
        xs = np.linspace(-4, 0.5, 20)
        Y1, Y2 = bayes.d_lj_different_B(xs, B_1, B_2, M_1, M_2)
        plt.plot(Y1, xs, 'k')

        xs = np.linspace(-4, 0.5, 20)
        Q1, Q2 = bayes.d_lj_different_B(xs, B_2, B_3, M_2, M_3)
        plt.plot(Q1, xs, 'm')
        xs = np.linspace(0.5, 4, 20)
        C1, C2 = bayes.d_lj_different_B(xs, B_1, B_3, M_1, M_3)
        plt.plot(C1, xs, 'y')

        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.show()

        p = probability(X_1, M_1, M_3, B_1, B_3, 0.5, 0.5)
        print("Вероятность ошибочной классификации: ", float(p))
        print("Относительная погрешность: ", math.sqrt(
            (1 - float(p)) / (200 * float(p))))
        print("Объем обуч выборки: ", int(
            (1 - float(p)) / (0.05 ** 2 * float(p))))
