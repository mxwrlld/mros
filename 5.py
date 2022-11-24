import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from sklearn import svm
import utils.constants as const
from utils.vector_generator import generate_norm_vector


def generate_linearly_separable_selection(
        N: int, B_1: np.ndarray, B_2: np.ndarray, M_1: np.ndarray, M_2: np.ndarray):
    is_linearly_separable = False
    x_1, x_2 = None, None
    while is_linearly_separable is False:
        x_1 = generate_norm_vector(N, B_1, M_1)
        x_2 = generate_norm_vector(N, B_1, M_2)
        x = np.concatenate((x_1, x_2), axis=1).T
        yldeal = np.zeros(shape=2*N)
        yldeal[N: 2*N] = 1
        clf = svm.LinearSVC()
        clf.fit(x, yldeal)
        def classify(x1, x2): return clf.coef_[
            0, 0] * x1 + clf.coef_[0, 1] * x2 + clf.intercept_
        classified0 = np.array(
            [0 if classify(x_1[0, i], x_1[1, i]) < 0 else 1 for i in range(N)])
        classified1 = np.array(
            [1 if classify(x_2[0, i], x_2[1, i]) > 0 else 0 for i in range(N)])
        print(classified0)
        print(classified1)
        if len(classified0[classified0 == 1]) == 0 and len(classified1[classified1 == 0]) == 0:
            is_linearly_separable = True
    return x_1, x_2


def generate_linearly_inseparable_selection(
        N: int, B_1: np.ndarray, B_2: np.ndarray, M_1: np.ndarray, M_2: np.ndarray):
    is_linearly_inseparable = False
    x_1, x_2 = None, None
    while is_linearly_inseparable is False:
        x_1 = generate_norm_vector(N, B_1, M_1)
        x_2 = generate_norm_vector(N, B_1, M_2)
        x = np.concatenate((x_1, x_2), axis=1).T
        yldeal = np.zeros(shape=2*N)
        yldeal[N: 2*N] = 1
        clf = svm.LinearSVC()
        clf.fit(x, yldeal)

        def classify(x1, x2): return clf.coef_[
            0, 0] * x1 + clf.coef_[0, 1] * x2 + clf.intercept_
        classified0 = np.array(
            [0 if classify(x_1[0, i], x_1[1, i]) < 0 else 1 for i in range(N)])
        classified1 = np.array(
            [1 if classify(x_2[0, i], x_2[1, i]) > 0 else 0 for i in range(N)])
        print(classified0)
        print(classified1)
        if len(classified0[classified0 == 1]) != 0 or len(classified1[classified1 == 0]) != 0:
            is_linearly_inseparable = True
    return x_1, x_2


if __name__ == "__main__":
    N = 50
    M_1, M_2 = const.M_1, const.M_2
    B_1, B_2 = const.B_1, const.B_2
    xs = np.linspace(-3, 3, N)
    Y_1 = generate_norm_vector(N, B_1, M_1)
    Y_2 = generate_norm_vector(N, B_1, M_2)
    X_1 = generate_norm_vector(N, B_1, M_1)
    X_2 = generate_norm_vector(N, B_2, M_2)

    x_1, x_2 = generate_linearly_inseparable_selection(N, B_1, B_1, M_1, M_2)
    z0, z1 = x_1, x_2
    x = np.concatenate((z0, z1), axis=1).T
    yldeal = np.zeros(shape=2*N)
    yldeal[N: 2*N] = 1
    clf = svm.LinearSVC()
    clf.fit(x, yldeal)

    x1, x2 = -3, 3

    def get_y(x): return -(clf.coef_[0, 0] *
                           x + clf.intercept_) / clf.coef_[0, 1]
    y1, y2 = get_y(x1), get_y(x2)

    plt.plot(z0[0], z0[1], color='red', marker='.', linestyle='none')
    plt.plot(z1[0], z1[1], color='blue', marker='.', linestyle='none')
    plt.plot([x1, x2], [y1, y2], color='green', marker='X', linestyle='solid')
    plt.show()

    yPredicted = clf.predict(x)
    yDif = np.abs(yldeal - yPredicted)
    Nerr = np.sum(yDif)
    yDif01 = yDif[0:N]
    yDif10 = yDif[N:2*N]
    N01 = np.sum(yDif01)
    N10 = np.sum(yDif10)
    print(Nerr/N)
    print(N01/N)
    print(N10/N)
