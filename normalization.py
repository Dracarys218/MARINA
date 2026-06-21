import numpy as np

class DynamicNorm:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.mean = None
        self.var = None

    def fit_train(self, train_data):
        self.mean = np.mean(train_data, axis=0)
        self.var = np.var(train_data, axis=0)

    def update(self, x):
        if self.mean is None or self.var is None:
            return x
        curr_mean = np.mean(x, axis=0)
        curr_var = np.var(x, axis=0)
        self.mean = (1 - self.alpha) * self.mean + self.alpha * curr_mean
        self.var = (1 - self.alpha) * self.var + self.alpha * curr_var
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    def static_norm(self, data):
        return (data - self.mean) / (np.sqrt(self.var) + 1e-8)