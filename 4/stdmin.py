import numpy as np
from numpy import linalg as la


class STDMin:
    def __init__(self, m: int, x_1: np.ndarray, x_2: np.ndarray):
        # m - объём обучающей выборки
        u, g = self.__generate_training_sample(m, x_1, x_2)
        self.w = self.__calc_w(u, g)

    # Вектор весовых коэффициентов
    def __calc_w(self, u: np.ndarray, g: np.ndarray):
        w = la.inv(u @ u.T) @ u @ g
        return w

    def __generate_training_sample(self, m: int, x_1: np.ndarray, x_2: np.ndarray):
        u, g = np.ndarray(shape=(3, m)), np.ndarray(shape=m)
        for i in range(m):
            if i < m / 2:
                u[0, i] = - x_1[0, i if i < x_1.shape[1] else i - x_1.shape[1]]
                u[1, i] = - x_1[1, i if i < x_1.shape[1] else i - x_1.shape[1]]
                u[2, i] = - 1
                g[i] = 1
            else:
                u[0, i] = x_2[0, i if i < x_1.shape[1] else i - x_1.shape[1]]
                u[1, i] = x_2[1, i if i < x_1.shape[1] else i - x_1.shape[1]]
                u[2, i] = 1
                g[i] = 1
        return u, g

    # Решающая функция
    def calc_decisive_function(self, x: np.ndarray):
        return self.w[0] * x[0, 0] + self.w[1] * x[1, 0] + self.w[2]

    def calc_decisive_functions(self, X: np.ndarray):
        return np.array([self.calc_decisive_function(np.array([[X[0, i]], [X[1, i]]]))
                         for i in range(X.shape[1])])

    # Решающая граница
    def calc_decisive_boundary(self, x: float):
        return (- self.w[0] * x - self.w[2]) / self.w[1]

    def calc_decisive_boundaries(self, xs: np.ndarray):
        return np.array([self.calc_decisive_boundary(xs[i]) for i in range(xs.shape[0])])

    def classify_vectors(self, X: np.ndarray, class_type: int, another_class_type: int):
        ds = self.calc_decisive_functions(X)
        if class_type == 0:
            return np.where(ds < 0, class_type, another_class_type)
        return np.where(ds > 0, class_type, another_class_type)
