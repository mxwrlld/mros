import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.stats import norm


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
    return np.array([generate_bin_vector(template_vector, p) for i in range(N)])


def calc_probabilities_matrix(vectors: np.ndarray): 
    return np.sum(vectors, axis=0) / vectors.shape[0]


def calc_math_expectation(p_0: np.ndarray, p_1: np.ndarray):
    m_0, m_1 = 0, 0
    for i in range(p_0.shape[0]):
        for j in range(p_0.shape[1]):
            m_0 += np.log(p_1[i][j] / (1 - p_1[i][j]) * (1 - p_0[i][j]) / p_0[i][j]) * p_0[i][j]
            m_1 += np.log(p_1[i][j] / (1 - p_1[i][j]) * (1 - p_0[i][j]) / p_0[i][j]) * p_1[i][j]
    return m_0, m_1 


def calc_std(p_0: np.ndarray, p_1: np.ndarray):
    std_0, std_1 = 0, 0
    for i in range(p_0.shape[0]):
        for j in range(p_0.shape[1]):
            std_0 += ((np.log(p_1[i][j] / (1 - p_1[i][j]) * (1 - p_0[i][j]) / p_0[i][j])) ** 2)  * p_0[i][j] * (1 - p_0[i][j])
            std_1 += ((np.log(p_1[i][j] / (1 - p_1[i][j]) * (1 - p_0[i][j]) / p_0[i][j])) ** 2) * p_1[i][j] * (1 - p_1[i][j])
    return np.sqrt(std_0), np.sqrt(std_1)


def calc_lyambda_tilda(p_0: np.ndarray, p_1: np.ndarray, P_0: float, P_1: float) -> float:
    lyambda_tilda = 0
    for i in range(0, p_0.shape[0]):
        for j in range(0, p_0.shape[1]):
            lyambda_tilda += np.log((1 - p_0[i][j]) / (1 - p_1[i][j]))
    lyambda_tilda += np.log(P_1 / P_0)
    return lyambda_tilda


if __name__ == "__main__":
    N = 200
    p = 0.3
    P_0 = P_1 = 0.5
    ya = get_YA()
    yu = get_YU()

    yas = generate_bin_vectors(ya, p, N)
    yus = generate_bin_vectors(yu, p, N)
    p_0, p_1 = calc_probabilities_matrix(yas), calc_probabilities_matrix(yus)
    m_0, m_1 = calc_math_expectation(p_0, p_1)
    std_0, std_1 = calc_std(p_0, p_1)
    lyambda_tilda = calc_lyambda_tilda(p_0, p_1, P_0, P_1)
    
    # print_bin_vectors("Я", ya, "Вектор на основе Я", yas[0])
    print("Ms: ", m_0, m_1)
    print("Stds: ", std_0, std_1)
    print(lyambda_tilda)
    fig = plt.figure()
    x = np.arange(-100, 100, 0.001)
    plt.plot(x, norm.pdf(x, m_0, std_0), color='red', label='yas')
    plt.plot(x, norm.pdf(x, m_1, std_1), color='green', label='yus')
    plt.axvline(x=lyambda_tilda, color='black')
    plt.legend()
    plt.ylim(0)
    plt.show()