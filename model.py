from typing import Dict, Tuple, Union

import flwr as fl
import numpy as np
from numpy import ndarray
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class Linear(fl.client.NumPyClient):
    def __init__(self, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray)\
            -> None:
        self.model = LinearRegression()

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        print(f'{self.x_train},\n{self.y_train},\n{self.x_test},\n{self.y_test}\n')

        # setting the initial parameter
        n_targets = 1
        n_features = 1
        self.model.coef_ = np.zeros((n_features))
        if self.model.fit_intercept:
            self.model.intercept_ = np.zeros((n_targets))

    def fit(self, parameters, config: Dict[str, Union[bool, bytes, float, int, str]])\
            -> Union[Tuple[any, any or float], int, Dict]:
        print(f'model coefficient: {self.model.coef_}')
        self.model.coef_ = parameters[0]
        print(f'server coefficient: {self.model.coef_}')
        if self.model.fit_intercept:
            self.model.intercept_ = parameters[1]

        self.model.fit(self.x_train, self.y_train)
        if 'rnd' in config.keys():
            print(f"Training finished for round {config['rnd']}")

        if self.model.fit_intercept:
            params = (self.model.coef_, self.model.intercept_)
        else:
            params = (self.model.coef_)

        print(f'coefficient after training: {self.model.coef_}\n')

        return params, len(self.x_train), {}

    def evaluate(self, parameters, config: Dict[str, Union[bool, bytes, float, int, str]])\
            -> Union[tuple or any, int, Dict[str, float or any or tuple]]:
        self.model.coef_ = parameters[0]
        if self.model.fit_intercept:
            self.model.intercept_ = parameters[1]

        loss = mean_squared_error(self.y_test, self.model.predict(self.x_test))
        accuracy = self.model.score(self.x_test, self.y_test)

        return loss, len(self.x_test), {'accuracy': accuracy}

    def get_parameters(self):
        if self.model.fit_intercept:
            params = (self.model.coef_, self.model.intercept_)
        else:
            params = (self.model.coef_)

        return params
