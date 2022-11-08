import numpy as np
from numpy import linalg as la


class Fisher:
    def __init__(self, M_1: np.ndarray, M_2: np.ndarray, B_1: np.ndarray, B_2: np.ndarray):
        self.w = self.__calc_w(M_1, M_2, B_1, B_2)
        self.w_n = self.__calc_w_n(M_1, M_2, B_1, B_2)

    # Вектор весовых коэффициентов
    def __calc_w(self, M_1: np.ndarray, M_2: np.ndarray, B_1: np.ndarray, B_2: np.ndarray):
        return la.inv(0.5 * (B_1 + B_2)) @ (M_2 - M_1)

    # Пороговое значение
    def __calc_w_n(self, M_1: np.ndarray, M_2: np.ndarray, B_1: np.ndarray, B_2: np.ndarray):
        std_1 = self.w.T @ B_1 @ self.w
        std_2 = self.w.T @ B_2 @ self.w
        w_n = (- ((M_2 - M_1).T @ la.inv(0.5 * (B_1 + B_2)) @
                  (std_1 * M_2 + std_2 * M_1)) / (std_1 + std_2))[0, 0]
        return w_n

    # Решающая функция
    def calc_decisive_function(self, x: np.ndarray):
        return self.w[0, 0] * x[0, 0] + self.w[1, 0] * x[1, 0] + self.w_n

    def calc_decisive_functions(self, X: np.ndarray):
        return np.array([self.calc_decisive_function(np.array([[X[0, i]], [X[1, i]]]))
                         for i in range(X.shape[1])])

    def calc_decisive_boundary(self, x: float):
        return (- self.w[0, 0] * x - self.w_n) / self.w[1, 0]

    def calc_decisive_boundaries(self, xs: np.ndarray):
        return np.array([self.calc_decisive_boundary(xs[i]) for i in range(xs.shape[0])])

    def classify_vectors(self, X: np.ndarray, class_type: int, another_class_type: int):
        ds = self.calc_decisive_functions(X)
        if class_type == 0:
            return np.where(ds < 0, class_type, another_class_type)
        return np.where(ds > 0, class_type, another_class_type)
