import numpy as np
# Подразумеваются векторы размерности 2
n = 2


def __calc_transform_matrix(B: np.ndarray):
    a_1_1 = np.sqrt(B[0, 0])
    a_1_2 = 0
    a_2_1 = B[0, 1] / np.sqrt(B[0, 0])
    a_2_2 = np.sqrt(B[1, 1] - (np.power(B[0, 1], 2) / B[1, 1]))
    return np.array([
        [a_1_1, a_1_2],
        [a_2_1, a_2_2]
    ])


def uniform2standard_normal_distribution(N):
    Zs = np.zeros(shape=(n, N))
    mean, deviation = 1 / 2, 1 / np.sqrt(12)
    len_of_uniform = 200
    for i in range(n):
        for j in range(N):
            uniform_distribution = np.random.uniform(size=len_of_uniform)
            _sum = np.sum(uniform_distribution)
            Zs[i, j] = (_sum - mean * len_of_uniform) / (deviation * np.sqrt(len_of_uniform))
    return Zs


def generate_norm_vector(N: int, B: np.ndarray, M: np.ndarray) -> np.ndarray:
    A = __calc_transform_matrix(B)
    R = uniform2standard_normal_distribution(N)
    X = np.dot(A, R) + M
    return X