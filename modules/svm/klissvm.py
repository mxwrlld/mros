import numpy as np
from modules.svm.svm import SVM


class KLISSVM(SVM):
    def __init__(self,
                 training_sample: np.ndarray, C: float = 1, kernel: str = None, params: dict = None):
        self.C = C
        self.kernel = kernel
        self.params = params
        super().__init__(
            training_sample
        )

    def calc_P(self, zs: np.ndarray):
        P = np.ndarray(shape=(self.m, self.m))

        for i in range(self.m):
            zs_i = np.array([[zs[0, i]], [zs[1, i]], [zs[2, i]]])
            for j in range(self.m):
                sign = zs[3, i] * zs[3, j]
                zs_j = np.array([[zs[0, j]], [zs[1, j]], [zs[2, j]]])
                k = self.get_kernel(zs_j, zs_i)
                P[i, j] = sign * k
        return P

    def get_discriminant_kernel(self, x):
        support_vectors_classes = self.support_vectors[3]
        sum_1 = np.sum(
            self.lyambdas[self.support_vectors_indexes] * support_vectors_classes *
            [self.get_kernel(self.support_vectors[:, i], x)
             for i in range(support_vectors_classes.shape[1])]
        )
        sum = 0
        for i in range(self.support_vectors.shape[1]):
            sum += support_vectors_classes[i] * self.lyambdas[self.support_vectors_indexes][:, i] * \
                self.get_kernel(self.support_vectors[i], x)
        return sum

    def calc_w_n(self):
        w_n = np.ndarray(shape=self.support_vectors.shape[0])
        for i in range(self.support_vectors.shape[0]):
            w_n[i] = self.get_discriminant_kernel(self.support_vectors[i])
        return w_n

    def get_kernel(self, x, y):
        d = self.params["d"]
        c = self.params["c"]
        return pow((x.T @ y)[0, 0] + c, d)

    def get_q(self):
        return np.ones(shape=(self.m, 1)) * (- 1)

    def get_G(self):
        return np.concatenate((np.eye(self.m) * -1, np.eye(self.m)), axis=0)

    def get_h(self):
        return np.concatenate((np.zeros((self.m,)), np.full((self.m,), self.C)), axis=0)

    def get_b(self):
        return np.zeros(shape=1)
