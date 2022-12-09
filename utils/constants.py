import numpy as np


M_1 = np.array([
    [-1],
    [1]
])
M_2 = np.array([
    [0],
    [-1]
])
M_3 = np.array([
    [2],
    [-1]
])
M_4 = np.array([
    [1],
    [-2]
])
M_5 = np.array([
    [1],
    [0]
])

B_1 = np.array([
    [0.4, 0.3],
    [0.3, 0.5]
])
B_2 = np.array([
    [0.3, 0],
    [0, 0.3]
])
B_3 = np.array([
    [0.87, -0.8],
    [-0.8, 0.95]
])
# Для ЛР №7
B_4 = np.array([
    [0.05, 0],
    [0, 0.05]
])
B_5 = np.array([
    [0.87, 0],
    [0, 0.95]
])

# Для генерации линейно разделимых выборок
M_1_ls = np.array([
    [-1],
    [1.5]
])
M_2_ls = np.array([
    [1.5],
    [-1.5]
])

# Для генерации линейно неразделимых выборок
M_1_lis = np.array([
    [-1],
    [1]
])
M_2_lis = np.array([
    [1.2],
    [2]
])