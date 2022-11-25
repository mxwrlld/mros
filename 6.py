import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
from sklearn import svm
import utils.constants as const
from utils.vector_generator import generate_norm_vector

# ==================== Параметры ==================== #
N = 100
B_1, B_2 = const.B_1, const.B_2
M_1_ls, M_2_ls = const.M_1_ls, const.M_2_ls
M_1_lis, M_2_lis = const.M_1_lis, const.M_2_lis
xs = np.linspace(-3, 3, N)


def get_vectors(generate: bool, save: bool):
    if generate:
        Y_1 = generate_norm_vector(N, B_1, M_1_ls)
        Y_2 = generate_norm_vector(N, B_1, M_2_ls)
        X_1 = generate_norm_vector(N, B_1, M_1_lis)
        X_2 = generate_norm_vector(N, B_2, M_2_lis)
        if save:
            np.savetxt("4/data/Y_1.txt", Y_1)
            np.savetxt("4/data/Y_2.txt", Y_2)
            np.savetxt("4/data/X_1.txt", X_1)
            np.savetxt("4/data/X_2.txt", X_2)
    else:
        Y_1 = np.loadtxt("data/Y_1.txt")
        Y_2 = np.loadtxt("data/Y_2.txt")
        X_1 = np.loadtxt("data/X_1.txt")
        X_2 = np.loadtxt("data/X_2.txt")
    return Y_1, Y_2, X_1, X_2


if __name__ == "__main__":
    config = {
        "generate": True,
        "save": False
    }

    # ==================== Синтез выборок двух классов ==================== #
    # 1. Синтезировать линейно разделимые выборки для двух классов двумерных случайных векторов в количестве N=100 в каждом классе
    S_1, S_2, IS_1, IS_2 = get_vectors(
        config["generate"], config["save"])


    # 2. Построить линейный классификатор по методу опорных векторов на выборке с линейно разделимыми классами. 
    # Использовать: 
    # - задачу (7) и метод решения квадратичных задач
    # - метод sklearn.svm.SVC библиотеки scikit-learn 
    # сопоставить решения из п.(б) с решением методом sklearn.svm.LinearSVC

    # 3. Построить линейный классификатор по SVM на выборке с линейно неразделимыми классами. Использовать для этого:
    # - задачу (12) и метод решения квадратичных задач. 
    #       Указать решения для C=1/10, 1, 10 и подобранно самостоятельно «лучшим коэффициентом».
    # - метод sklearn.svm.SVC библиотеки scikit-learn

    # 4. Построить классификатор по SVM, разделяющий линейно неразделимые классы. 
    # Использовать: 
    # - задачу (14) и метод решения квадратичных задач,
    #       Исследовать решение для различных значений параметра C=1/10, 1, 10 и различных ядер из таблицы 1
    # - метод sklearn.svm.SVC.

    z0, z1 = S_1, S_2
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
