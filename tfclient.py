import flwr as fl
import numpy as np

import textwrap

from typing import Dict, Tuple, Union
from numpy import ndarray
# from sklearn.metrics import categorical_accuracy

import pandas as pd
import tensorflow as tf

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, InputLayer
from create_dataset import create_dataset


class TFclient(fl.client.NumPyClient):
    def __init__(self, x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray)\
            -> None:
        self.model = Sequential([
            InputLayer(input_shape=(1,)),
            Dense(1)
        ])

        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.01),
            loss='mean_squared_error', metrics=['categorical_accuracy'])

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def fit(self, parameters, config: Dict[str, Union[bool, bytes, float, int, str]])\
            -> Union[Tuple[any, any or float], int, Dict]:
        self.model.set_weights(parameters)

        if 'batch_size' in config.keys():
            batch_size: int = config["batch_size"]
        else:
            batch_size = 32
        if 'local_epochs' in config.keys():
            epochs: int = config["local_epochs"]
        else:
            epochs = 1

        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
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
            "categorical_accuracy": history.history["categorical_accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_categorical_accuracy": history.history["val_categorical_accuracy"][0],
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
        loss, accuracy = self.model.evaluate(
            self.x_test, self.y_test, 32, steps=steps)

        num_examples_test = len(self.x_test)

        return loss, num_examples_test, {"accuracy": accuracy}

    def get_parameters(self):
        raise Exception(
            "Not implemented (server-side parameter initialization)")


if __name__ == '__main__':
    X, Y = create_dataset(nb=20)
    (x_train, x_test, y_train, y_test) = train_test_split(
        np.array(X), np.array(Y[0]), train_size=0.75)

    client = TFclient(x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("localhost:8080", client=client)
