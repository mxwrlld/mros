import numpy as np
import qpsolvers
from qpsolvers import solve_qp


class LSSVM:
    def __init__(self, training_sample: np.ndarray):
        # m - объём обучающей выборки, training sample == zs
        self.m = training_sample.shape[1]
        print("available_solvers: ", qpsolvers.available_solvers)
        lyambdas = solve_qp(
            self.__calc_P(training_sample),
            self.__get_q(),
            self.__get_G(),
            self.__get_h(),
            self.__calc_A(training_sample),
            self.__get_b(),
            solver='cvxopt'
        )
        # Получение индексов векторов, претендующих на опорные
        support_vectors_indexes = np.where(lyambdas > 0.000001, True, False)
        self.support_vectors = training_sample[:, support_vectors_indexes]
        self.w = self.__calc_w(
            self.support_vectors, lyambdas[support_vectors_indexes])
        self.w_n = self.__calc_w_n(self.support_vectors)

    def __calc_P(self, zs: np.ndarray):
        P = np.ndarray(shape=(self.m, self.m))

        for i in range(self.m):
            zs_i = np.array([[zs[0, i]], [zs[1, i]], [zs[2, i]]])
            for j in range(self.m):
                # sign = r_i * r_j
                sign = zs[3, i] * zs[3, j]
                zs_j = np.array([[zs[0, j]], [zs[1, j]], [zs[2, j]]])
                P[i, j] = sign * (zs_j.T @ zs_i)
        return P

    def __get_q(self):
        return np.ones(shape=(self.m, 1)) * (- 1)

    def __get_G(self):
        return np.eye(self.m) * -1

    def __get_h(self):
        return np.zeros((self.m))

    def __calc_A(self, zs: np.ndarray):
        A = np.array([zs[3, i] for i in range(self.m)])
        return A

    def __get_b(self):
        return np.zeros(shape=1)

    # Расчёт матрицы весовых коэффициентов через двойственные переменные
    def __calc_w(self, support_vectors, support_vectors_lyambdas):
        w = np.sum(support_vectors[0:2] *
                   support_vectors[3] * support_vectors_lyambdas, axis=1)
        return np.reshape(w, newshape=(2, 1))

    def __calc_w_n(self, support_vectors):
        if support_vectors.shape[1] != 0:
            return (support_vectors[3, -1] - (self.w.T @ support_vectors[0:2, -1]))[0]
        return None
