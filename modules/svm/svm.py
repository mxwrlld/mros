import numpy as np
from qpsolvers import solve_qp


class SVM:
    def __init__(self, training_sample, threshold):
        # m - объём обучающей выборки, training sample == zss
        self.m = training_sample.shape[1]
        self.lyambdas = solve_qp(
            self.calc_P(training_sample),
            self.get_q(),
            self.get_G(),
            self.get_h(),
            self.calc_A(training_sample),
            self.get_b(),
            solver='cvxopt'
        )
        # Получение индексов векторов, претендующих на опорные
        self.support_vectors_indexes = np.where(
            self.lyambdas > threshold, True, False)
        self.support_vectors = training_sample[:, self.support_vectors_indexes]
        self.w = self.calc_w(
            self.support_vectors, self.lyambdas[self.support_vectors_indexes])
        self.w_n = self.calc_w_n()
        # print("Support vectors count: ", self.support_vectors.shape[1])
        # print("Support vectors: ", self.support_vectors)
        # print("W: ", self.w)
        # print("W_n: ", self.w_n)

    def calc_P(self, zs: np.ndarray):
        P = np.ndarray(shape=(self.m, self.m))

        for i in range(self.m):
            zs_i = np.array([[zs[0, i]], [zs[1, i]]])
            for j in range(self.m):
                # sign = r_i * r_j
                sign = zs[2, i] * zs[2, j]
                zs_j = np.array([[zs[0, j]], [zs[1, j]]])
                P[i, j] = sign * (zs_j.T @ zs_i)
        return P

    def calc_A(self, zs: np.ndarray):
        A = np.array([zs[2, i] for i in range(self.m)])
        return A

    # Расчёт матрицы весовых коэффициентов через двойственные переменные
    def calc_w(self, support_vectors, support_vectors_lyambdas):
        w = np.sum(support_vectors[0:2] *
                   support_vectors[2] * support_vectors_lyambdas, axis=1)
        return np.reshape(w, newshape=(2, 1))

    # Расчёт порогового значения
    def calc_w_n(self):
        if self.support_vectors.shape[1] != 0:
            return np.mean(
                self.support_vectors[2] - self.w.T @ self.support_vectors[0:2])
        return None
