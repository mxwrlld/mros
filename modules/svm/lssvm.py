import numpy as np
from svm import SVM

class LSSVM(SVM):
    def __init__(self, training_sample: np.ndarray):
        super().__init__(
            training_sample
            )

    def get_q(self):
        return np.ones(shape=(self.m, 1)) * (- 1)

    def get_G(self):
        return np.eye(self.m) * -1

    def get_h(self):
        return np.zeros((self.m))

    def get_b(self):
        return np.zeros(shape=1)