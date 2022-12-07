import numpy as np

lp = np.loadtxt(
    "B:\Study\9_semestr\МРО (Мясников. Баврина)\mros\data\Ps\L_P.txt")
lm = np.loadtxt(
    "B:\Study\9_semestr\МРО (Мясников. Баврина)\mros\data\Ps\L_M.txt")

dif = lp - lm
print(dif)
print(dif.max(), dif.min())
