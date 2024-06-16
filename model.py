import numpy as np
import matplotlib as mp
import pickle

class MyLinearRegression:
    def __init__(self):
        self.unnormalized_independent = []
        self.independent = []
        self.dependent = []
        self.slope = 0
        self.gradient = 0
        self.best_slope = 0
        self.best_gradient = 0
        self.best_errors = []
        self.learning_rate = 0.0005

    def _calculate_gradients(self):
        predictions = self.predict(self.independent)
        errors = predictions - self.dependent
        return np.mean(errors), np.mean(errors * self.independent)

    def normilized_dataset(self):
        result = (self.independent - np.mean(self.independent)) / np.std(self.independent)
        return result

    def predict(self, independent):
        result = self.gradient + (self.slope * independent)
        return result

    def set_best_coefficients(self):
        predictions = self.predict(self.independent)
        errors = predictions - self.dependent
        if (np.sum(np.square(errors)) < np.sum(np.square(self.best_errors))):
            self.best_errors = errors
            self.best_slope = self.slope
            self.best_gradient = self.gradient

    def calculate_original_coefficients(self):
        mu = np.mean(self.unnormalized_independent)
        sigma = np.std(self.unnormalized_independent)
        self.slope = self.best_slope / sigma
        self.gradient = self.best_gradient - (self.best_slope * mu / sigma)

    def save_model(self):
        try:
            with open('model.pkl', 'wb') as file:
                pickle.dump(self, file)
        except:
            raise RuntimeError("Error while saving model")

    def fit(self, independent, dependent):
        self.independent = np.array(independent.values)
        self.unnormalized_independent = np.array(independent.values)
        self.dependent = np.array(dependent.values)
        self.independent = self.normilized_dataset()
        try:
            if len(independent) != len(dependent):
                raise AttributeError("Length of arrays of values should be equal")
        except:
            raise AttributeError("Arguments should be iterable")
        self.best_errors = np.full(len(self.independent), np.inf)
        learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 5]
        for rate in learning_rates:
            self.learning_rate = rate
            for _ in range(1000):
                tmp_gradient, tmp_slope = self._calculate_gradients()
                self.gradient -= self.learning_rate * tmp_gradient
                self.slope -= self.learning_rate * tmp_slope
                self.set_best_coefficients()
        self.calculate_original_coefficients()
        self.save_model()
        self.draw_plot()


    def draw_plot(self):
        mp.pyplot.scatter(self.unnormalized_independent, self.dependent)
        mp.pyplot.plot(self.unnormalized_independent, self.predict(self.unnormalized_independent))
        mp.pyplot.show()
