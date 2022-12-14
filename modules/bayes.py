import math
import numpy as np
from numpy import linalg as la


def calc_decisive_function(P_l: float, B_l: np.ndarray, M_l: np.ndarray, x: np.ndarray):
    return np.log(P_l) - np.log(np.sqrt(la.det(B_l))) - 0.5 * ((x - M_l).T @ la.inv(B_l) @ (x - M_l))


def classify_vectors(X: np.ndarray, Bs: np.ndarray, Ms: np.ndarray, Ps: np.ndarray):
    res = np.ndarray(shape=X.shape[1])
    ds = np.ndarray(shape=Bs.shape[0])
    for i in range(X.shape[1]):
        for j in range(Bs.shape[0]):
            ds[j] = calc_decisive_function(
                Ps[j], Bs[j], Ms[j], np.array([[X[0, i]], [X[1, i]]]))
        res[i] = np.argmax(ds)
    return res


def d_lj_equal_B(xs: np.ndarray, B: np.ndarray, M_i: np.ndarray, M_j: np.ndarray, P_i: float, P_j: float):
    l = np.dot((M_j - M_i).T, la.inv(B))
    r = (- 0.5) * np.dot(np.dot((M_j + M_i).T, la.inv(B)), (M_j - M_i))[0, 0]
    log_appr = np.log(P_i / P_j)
    ys = [(-1) * (l[0, 0] * x + r + log_appr) / l[0, 1] for x in xs]
    return np.array(ys)


def d_lj_different_B(xs: np.ndarray, B_i: np.ndarray, B_j: np.ndarray, M_i: np.ndarray, M_j: np.ndarray):
    c_dif = la.inv(B_j) - la.inv(B_i)
    a_ = c_dif[0, 0]
    b_ = c_dif[1, 0] + c_dif[0, 1]
    c_ = c_dif[1, 1]
    s = 2 * (np.dot(M_i.T, la.inv(B_i)) - np.dot(M_j.T, la.inv(B_j)))
    d_ = s[0, 0]
    e_ = s[0, 1]
    f_ = math.log(la.det(B_i) / la.det(B_j)) - np.dot(np.dot(M_i.T, la.inv(B_i)), M_i) + np.dot(
        np.dot(M_j.T, la.inv(B_j)), M_j)
    Y1, Y2 = np.copy(xs), np.copy(xs)
    for i in range(len(xs)):
        x = xs[i]
        a = a_
        b = b_ * x + d_
        c = c_ * (x ** 2) + e_ * x + f_
        D = b ** 2 - 4 * a * c
        y1 = ((- b - np.sqrt(D)) / (2 * a))[0, 0]
        y2 = ((- b + np.sqrt(D)) / (2 * a))[0, 0]
        Y1[i], Y2[i] = y1, y2
    return Y1, Y2


def calc_decisive_function(
        x: np.ndarray, B: np.ndarray, M: np.ndarray, P: float):
    return math.log(P) - math.log(np.sqrt(la.det(B))) - \
        0.5 * (x - M).T @ la.inv(B) @ (x - M)


def bayes_classification():
    return 0


def calc_class_of_vector(
        x: np.ndarray, Bs: list, Ms: list):
    count_of_classes = len(Bs)
    ds = [
        calc_decisive_function(
            x.reshape((2, 1)), Bs[i], Ms[i], 1 / count_of_classes)
        for i in range(count_of_classes)
    ]
    return np.argmax(ds)


def bayes(test_sample: np.ndarray, Bs: list, Ms: list):
    vcalc_class_of_vector = np.vectorize(
        calc_class_of_vector,
        signature='(n) -> ()',
        excluded=[1, 2]
    )

    classification_res = vcalc_class_of_vector(test_sample.T, Bs, Ms)
    return classification_res
