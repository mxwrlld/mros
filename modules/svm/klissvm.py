import numpy as np
from modules.svm.svm import SVM


class KLISSVM(SVM):
    def __init__(self,
                 training_sample: np.ndarray, C: float = 1, kernel: str = None, params: dict = None):
        self.C = C
        self.kernel = kernel
        self.params = params
        super().__init__(
            training_sample,
            threshold=0.00001
        )

    def calc_P(self, zs: np.ndarray):
        P = np.ndarray(shape=(self.m, self.m))

        for i in range(self.m):
            zs_i = np.array([[zs[0, i]], [zs[1, i]]])
            for j in range(self.m):
                sign = zs[2, i] * zs[2, j]
                zs_j = np.array([[zs[0, j]], [zs[1, j]]])
                k = self.get_kernel(zs_j, zs_i)
                P[i, j] = sign * k
        return P

    def get_discriminant_kernel(self, x):
        return np.sum(
            self.lyambdas[self.support_vectors_indexes]
            * self.support_vectors[2, :]
            * np.array([self.get_kernel(self.support_vectors[0:2, i], x)
                        for i in range(self.support_vectors.shape[1])])
        )

    def calc_w_n(self):
        w_n = np.ndarray(shape=self.support_vectors.shape[1])
        for i in range(self.support_vectors.shape[1]):
            w_n[i] = self.get_discriminant_kernel(self.support_vectors[0:2, i])
        w_n = np.mean(self.support_vectors[2] - w_n)
        return w_n

    def get_kernel(self, x, y):
        if self.kernel == 'poly':
            d = self.params["d"]
            c = self.params["c_p"]
            return pow((x.T @ y) + c, d)
        if self.kernel == 'sigmoid':
            c = self.params["c_s"]
            g = self.params["g_s"]
            return np.tanh(g * (x.T @ y) + c)
        if self.kernel == 'rbf':
            g = self.params["g_r"]
            return np.exp(-g * np.sum((x - y) ** 2))
        if self.kernel == 'rbf_gauss':
            g = self.params["g_r_gauss"]
            return np.exp(-g * np.sum((x - y) ** 2))
        return None

    def get_q(self):
        return np.ones(shape=(self.m, 1)) * (- 1)

    def get_G(self):
        return np.concatenate((np.eye(self.m) * -1, np.eye(self.m)), axis=0)

    def get_h(self):
        return np.concatenate((np.zeros((self.m,)), np.full((self.m,), self.C)), axis=0)

    def get_b(self):
        return np.zeros(shape=1)
