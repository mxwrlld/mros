import math
import numpy as np
from numpy import linalg as la


def calc_parsen_kernel(x: np.ndarray, x_i: np.ndarray, B: np.ndarray, e_power_multiplier, const):
    power = e_power_multiplier * (x - x_i) @ np.linalg.inv(B) @ (x - x_i)
    return const * np.exp(power)


def parsen_classification(x, train_vectors_by_classes, Bs, count_of_classes, count_of_train_vectors, vcalc_parsen_kernel):
    f = np.zeros(count_of_classes)
    P = np.zeros(count_of_classes)
    k = 0.3

    for j in range(count_of_classes):
        vectors_by_class = train_vectors_by_classes[j]
        _class_dim = vectors_by_class.shape[1]  # -- N
        h = _class_dim ** (- k / 2)   # -- (11)
        const = (2 * np.pi) * (h ** (- 2)) * \
            (np.linalg.det(Bs[j]) ** (- 0.5))
        exp_sequence = []
        e_power_multiplier = (- 0.5 * (h ** (- 2)))
        exp_sequence = vcalc_parsen_kernel(
            x, vectors_by_class.T, Bs[j], e_power_multiplier, const)
        f[j] = np.average(exp_sequence)
        P[j] = _class_dim

    return np.argmax((P / count_of_train_vectors) * f)


def parsen(train_vectors_by_classes: np.ndarray, test_sample: np.ndarray, Bs: np.ndarray):
    classification_res = np.ndarray(shape=test_sample.shape[1])
    count_of_classes = len(train_vectors_by_classes)
    count_of_train_vectors = np.sum([train_vectors_by_classes[i].shape[1]
                                     for i in range(count_of_classes)])

    vcalc_parsen_kernel = np.vectorize(
        calc_parsen_kernel,
        signature='(n), (), ()->()',
        excluded=[0, 2]
    )
    vparsen_classification = np.vectorize(
        parsen_classification,
        signature='(n), (), (), ()->()',
        excluded=[1, 2]
    )

    classification_res = vparsen_classification(
        test_sample.T, train_vectors_by_classes, Bs, count_of_classes, count_of_train_vectors, vcalc_parsen_kernel)

    return classification_res
