import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


def get_YA() -> np.ndarray:
    return np.array([[0, 0, 0, 1, 1, 1, 1, 1, 0],
                     [0, 0, 1, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 0],
                     [0, 0, 0, 0, 1, 0, 0, 1, 0],
                     [0, 0, 0, 1, 0, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 0, 0, 1, 0]])


def get_YU() -> np.ndarray:
    return np.array([[1, 0, 0, 0, 0, 1, 1, 0, 0],
                     [1, 0, 0, 0, 1, 0, 0, 1, 0],
                     [1, 0, 0, 1, 0, 0, 0, 0, 1],
                     [1, 0, 0, 1, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 0, 0, 0, 0, 1],
                     [1, 0, 0, 1, 0, 0, 0, 0, 1],
                     [1, 0, 0, 1, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 1, 0, 0, 1, 0],
                     [1, 0, 0, 0, 0, 1, 1, 0, 0]])


def print_bin_vectors(f_name, f, s_name, s):
    fig = plt.figure(frameon=False)
    fig.add_subplot(1, 2, 1)
    plt.title(f_name)
    plt.imshow(1 - f, cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.title(s_name)
    plt.imshow(1 - s, cmap='gray')
    plt.show()


def generate_bin_vector(template_vector: np.ndarray, p: float) -> np.ndarray:
    bin_vector = np.ndarray(shape=template_vector.shape)
    us = np.random.uniform(0, 1, size=template_vector.shape)
    bin_vector = np.where(us > p, template_vector, 1 - template_vector)
    return bin_vector


def generate_bin_vectors(template_vector: np.ndarray, p: float, N: int) -> np.ndarray:
    vectors = np.array([generate_bin_vector(template_vector, p)
                       for i in range(N)])
    return vectors

def calc_probability(vector): 
    return np.sum(vector) / (vector.shape[1] * vector.shape[0])


if __name__ == "__main__":
    N = 5
    p = 0.3
    YA = get_YA()
    YU = get_YU()

    vector_ya = generate_bin_vector(YA, p)
    print(calc_probability(vector_ya))
    print_bin_vectors("Я", YA, "Вектор на основе Я", vector_ya)
