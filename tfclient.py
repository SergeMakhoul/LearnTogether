import textwrap
import sys
from typing import Dict, Tuple, Union

import flwr as fl
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import ndarray
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizers import gradient_descent_v2
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dense, InputLayer

from dataset.create_dataset import create_dataset


class TFclient(fl.client.NumPyClient):
    def __init__(self, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray)\
            -> None:
        self.model = Sequential([
            InputLayer(input_shape=(1,)),
            Dense(1)
        ])

        self.model.compile(
            optimizer=gradient_descent_v2.SGD(learning_rate=0.01),
            loss='mean_squared_error')

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def fit(self, parameters, config: Dict[str, Union[bool, bytes, float, int, str]])\
            -> Union[Tuple[any, any or float], int, Dict]:
        self.model.set_weights(parameters)

        batch_size: int = config["batch_size"] if 'batch_size' in config.keys(
        ) else 32
        epochs: int = config["local_epochs"] if 'local_epochs' in config.keys(
        ) else 1

        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
        )

        parameters_prime = self.model.get_weights()

        print(textwrap.dedent(f"""
            ****************

            {parameters_prime}

            {history.history}

            ****************
        """))

        num_examples_train = len(self.x_train)

        results = {
            "loss": history.history["loss"][0],
            "val_loss": history.history["val_loss"][0],
        }

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config: Dict[str, Union[bool, bytes, float, int, str]])\
            -> Union[tuple or any, int, Dict[str, float or any or tuple]]:
        self.model.set_weights(parameters)

        # Get config values
        if 'val_steps' in config.keys():
            steps: int = config["val_steps"]
        else:
            steps = 5

        # Evaluate global model parameters on the local test data and return results
        loss = self.model.evaluate(
            self.x_test, self.y_test, len(self.x_test) // steps, steps=steps)

        num_examples_test = len(self.x_test)

        return loss, num_examples_test, {}

    def get_parameters(self):
        raise Exception(
            "Not implemented (server-side parameter initialization)")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        data = pd.read_csv('dataset/dataset.csv')[100*(n-1): 100*n]
    else:
        data = pd.read_csv('dataset/dataset.csv').sample(100)

    # X, Y = create_dataset()

    X = data['X']
    Y = data['Y']

    (x_train, x_test, y_train, y_test) = train_test_split(
        X.to_numpy(), Y.to_numpy(), train_size=0.75)

    client = TFclient(x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("localhost:8080", client=client)
