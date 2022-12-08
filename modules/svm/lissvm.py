import numpy as np
from modules.svm.svm import SVM


class LISSVM(SVM):
    def __init__(self,
                 training_sample: np.ndarray, C: float = 1):
        self.C = C
        super().__init__(
            training_sample,
            threshold=1e-4
        )

    def get_q(self):
        return np.ones(shape=(self.m, 1)) * (- 1)

    def get_G(self):
        return np.concatenate((np.eye(self.m) * -1, np.eye(self.m)), axis=0)

    def get_h(self):
        return np.concatenate((np.zeros((self.m,)), np.full((self.m,), self.C)), axis=0)

    def get_b(self):
        return np.zeros(shape=1)
