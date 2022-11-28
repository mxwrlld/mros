import numpy as np
import qpsolvers
from qpsolvers import solve_qp

class SVM:
    def __init__(self, training_sample):
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
        support_vectors_indexes = np.where(self.lyambdas > 0.0000001, True, False)
        self.support_vectors = training_sample[:, support_vectors_indexes]
        self.w = self.calc_w(
            self.support_vectors, self.lyambdas[support_vectors_indexes])
        self.w_n = self.calc_w_n(self.support_vectors)
    
    def calc_P(self, zs: np.ndarray):
        P = np.ndarray(shape=(self.m, self.m))

        for i in range(self.m):
            zs_i = np.array([[zs[0, i]], [zs[1, i]], [zs[2, i]]])
            for j in range(self.m):
                # sign = r_i * r_j
                sign = zs[3, i] * zs[3, j]
                zs_j = np.array([[zs[0, j]], [zs[1, j]], [zs[2, j]]])
                P[i, j] = sign * (zs_j.T @ zs_i)
        return P

    def calc_A(self, zs: np.ndarray):
        A = np.array([zs[3, i] for i in range(self.m)])
        return A

    # Расчёт матрицы весовых коэффициентов через двойственные переменные
    def calc_w(self, support_vectors, support_vectors_lyambdas):
        w = np.sum(support_vectors[0:2] *
                   support_vectors[3] * support_vectors_lyambdas, axis=1)
        return np.reshape(w, newshape=(2, 1))
    
    # Расчёт порогового значения
    def calc_w_n(self, support_vectors):
        if support_vectors.shape[1] != 0:
            return (support_vectors[3, -1] - (self.w.T @ support_vectors[0:2, -1]))[0]
        return None


class LSSVM(SVM):
    def __init__(self, training_sample: np.ndarray):
        super().__init__(
            training_sample
            )

    def get_q(self):
        return np.ones(shape=(self.m, 1)) * (- 1)

    def get_G(self):
        return np.eye(self.m) * -1

    def get_h(self):
        return np.zeros((self.m))

    def get_b(self):
        return np.zeros(shape=1)


class LISSVM(SVM):
    def __init__(self, training_sample: np.ndarray, C: float=1):
        # m - объём обучающей выборки, training sample == zs
        self.C = C
        super().__init__(
            training_sample
            )

    def get_q(self):
        return np.ones(shape=(self.m, 1)) * (- 1)

    def get_G(self):
        return np.concatenate((np.eye(self.m) * -1, np.eye(self.m)), axis=0)

    def get_h(self):
        return np.concatenate((np.zeros((self.m,)), np.full((self.m,), self.C)), axis=0)

    def get_b(self):
        return np.zeros(shape=1)