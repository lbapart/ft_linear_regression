import numpy as np
from sklearn.preprocessing import StandardScaler

class MyLinearRegression:
    def __init__(self):
        self.independent = []
        self.dependent = []
        self.slope = np.random.randn()
        self.gradient = np.random.randn()
        self.learning_rate = 0.0015

    def _calculate_gradients(self):
        predictions = self.predict(self.independent)
        errors = predictions - self.dependent
        return np.mean(errors), np.mean(errors * self.independent)


    def predict(self, independent):
        result = self.gradient + (self.slope * independent)
        return result

    def fit(self, independent, dependent):
        self.independent = np.array(independent.values)
        self.dependent = np.array(dependent.values)
        try:
            if len(independent) != len(dependent):
                raise AttributeError("Length of arrays of values should be equal")
        except:
            raise AttributeError("Arguments should be iterable")
        for _ in range(100):
            tmp_gradient, tmp_slope = self._calculate_gradients()
            self.gradient -= self.learning_rate * tmp_gradient
            self.slope -= self.learning_rate * tmp_slope
            print(f"Gradient: {self.gradient}")
            print(f"Slope: {self.slope}")
        print(self.slope, self.gradient)