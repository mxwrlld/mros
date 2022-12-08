import numpy as np

a = np.load("data\\test\wnm.npy")
b = np.load("data\\test\wnp.npy")
dif = a - b
print(dif)
# print(dif.max(), dif.min())
