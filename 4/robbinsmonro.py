import numpy as np
from numpy import linalg as la


class RobbinsMonro:
    def __init__(self, m: int, betta: float, initial_w: float, x_1: np.ndarray, x_2: np.ndarray):
        # m - число итераций
        self.m = m
        self.betta = betta
        self.initial_w = initial_w
        alphas = self.__generate_alphas()
        # train_sample:
        #   0 : "классика",
        #   1 : "классика с дополнительным переупорядочиванием"
        # method:
        #   0 : "классика - долгая сходимость"
        #   1 : "корректировка - быстрая"
        calc_w_mode = {"train_sample": 0, "method": 1}
        if 0 == calc_w_mode["train_sample"]:
            zs, rs = self.__generate_training_sample(x_1, x_2)
            if 0 == calc_w_mode["method"]:
                self.w = self.__calc_w(zs, rs, alphas)
            else:
                self.w = self.__calc_w_correction(zs, rs)
        else:
            zs = self.__generate_training_sample_2(x_1, x_2)
            if 0 == calc_w_mode["method"]:
                self.w = self.__calc_w_2(zs, alphas)
            else:
                self.w = self.__calc_w_correction(zs, rs)
        self.w_length = self.w.shape[1]

    def __generate_training_sample(self, x_1: np.ndarray, x_2: np.ndarray):
        length = x_1.shape[1]
        zs, rs = np.ndarray(shape=(3, self.m)), np.ndarray(shape=self.m)
        for i in range(self.m):
            if i % 2 == 0:
                zs[0, i] = x_1[0, int((i / 2) % length)]
                zs[1, i] = x_1[1, int((i / 2) % length)]
                rs[i] = - 1
            else:
                zs[0, i] = x_2[0, int((i - 1 / 2) % length)]
                zs[1, i] = x_2[1, int((i - 1 / 2) % length)]
                rs[i] = 1
            zs[2, i] = 1
        return zs, rs

    def __generate_training_sample_2(self, x_1: np.ndarray, x_2: np.ndarray):
        length = x_1.shape[1]
        zs = np.ndarray(shape=(4, self.m))
        for i in range(self.m):
            if i % 2 == 0:
                zs[0, i] = x_1[0, int((i / 2) % length)]
                zs[1, i] = x_1[1, int((i / 2) % length)]
                zs[3, i] = - 1
            else:
                zs[0, i] = x_2[0, int((i - 1 / 2) % length)]
                zs[1, i] = x_2[1, int((i - 1 / 2) % length)]
                zs[3, i] = 1
            zs[2, i] = 1
        # Дополнительное переупорядочивание
        np.random.default_rng().shuffle(zs, axis=1)
        return zs

    def __generate_alpha(self, i: int):
        return 1 / (i ** self.betta)

    def __generate_alphas(self):
        alphas = np.array([1 / (i ** self.betta)
                          for i in range(1, self.m + 1)])
        return alphas

    def __calc_w_correction(self, zs: np.ndarray, rs: np.ndarray):
        w = np.ones(shape=(3, 1))
        ws = np.ones(shape=(3, 1)) * self.initial_w
        alpha_count = 1
        for i in range(self.m - 1):
            dec = ws[0, -1].T * zs[0, i] + ws[1, -1].T * zs[1, i] + ws[2, -1]
            if (dec < 0 and rs[i] > 0) or (dec > 0 and rs[i] < 0):
                alpha = self.__generate_alpha(alpha_count)
                alpha_count += 1
                w[0, 0] = ws[0, -1] + alpha * zs[0, i] * (rs[i] - dec)
                w[1, 0] = ws[1, -1] + alpha * zs[1, i] * (rs[i] - dec)
                w[2, 0] = ws[2, -1] + alpha * zs[2, i] * (rs[i] - dec)
                ws = np.append(ws, w, axis=1)
        return ws

    def __calc_w(self, zs: np.ndarray, rs: np.ndarray, alphas: np.ndarray):
        w = np.ones(shape=(3, self.m)) * self.initial_w
        for i in range(self.m - 1):
            w[0, i + 1] = w[0, i] + alphas[i] * \
                zs[0, i] * (rs[i] - w[0, i].T * zs[0, i])
            w[1, i + 1] = w[1, i] + alphas[i] * \
                zs[1, i] * (rs[i] - w[1, i].T * zs[1, i])
            w[2, i + 1] = w[2, i] + alphas[i] * \
                zs[2, i] * (rs[i] - w[2, i].T * zs[2, i])
        return w

    def __calc_w_2(self, zs: np.ndarray, alphas: np.ndarray):
        w = np.ones(shape=(3, self.m)) * self.initial_w
        for i in range(self.m - 1):
            w[0, i + 1] = w[0, i] + alphas[i] * \
                zs[0, i] * (zs[3, i] - w[0, i].T * zs[0, i])
            w[1, i + 1] = w[1, i] + alphas[i] * \
                zs[1, i] * (zs[3, i] - w[1, i].T * zs[1, i])
            w[2, i + 1] = w[2, i] + alphas[i] * \
                zs[2, i] * (zs[3, i] - w[2, i].T * zs[2, i])
        return w

    def calc_decisive_function(self, x: np.ndarray, index: int):
        return self.w[0, index] * x[0, 0] + self.w[1, index] * x[1, 0] + self.w[2, index]

    def calc_decisive_functions(self, X: np.ndarray, index: int):
        return np.array([self.calc_decisive_function(np.array([[X[0, i]], [X[1, i]]]), index)
                         for i in range(X.shape[1])])

    # Решающая граница
    def calc_decisive_boundary(self, x: float, index: int):
        return (- self.w[0, index] * x - self.w[2, index]) / self.w[1, index]

    def calc_decisive_boundaries(self, xs: np.ndarray, index: int = None):
        index = index if index is not None else self.w_length - 1
        return np.array([self.calc_decisive_boundary(xs[i], index) for i in range(xs.shape[0])])

    def classify_vectors(
            self, X: np.ndarray, class_type: int, another_class_type: int, index: int = None):
        index = index if index is not None else self.w_length - 1
        ds = self.calc_decisive_functions(X, index)
        if class_type == 0:
            return np.where(ds < 0, class_type, another_class_type)
        return np.where(ds > 0, class_type, another_class_type)
