import json
import os
import sys
import textwrap
from argparse import ArgumentParser
from typing import Dict, Tuple, Union

import flwr as fl
import pandas as pd
from numpy import ndarray
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dense, InputLayer
from tensorflow.python.keras.optimizers import gradient_descent_v2

from utils import create_dataset, save_history

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class TFclient(fl.client.NumPyClient):
    def __init__(
            self,
            x_train: ndarray,
            y_train: ndarray,
            x_test: ndarray = None,
            y_test: ndarray = None,
            num: int = 0
    ) -> None:
        self.model = Sequential([
            InputLayer(input_shape=(1,)),
            Dense(1)
        ])

        self.model.compile(
            optimizer=gradient_descent_v2.SGD(
                learning_rate=config['model']['learning_rate']
            ),
            loss='mean_squared_error')

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        self.name = f'client{num}'

        self.history = {
            'loss': [],
            'val_loss': [],
            'weights': []
        }

    def __save_history(self):
        save_history(self.name, self.history)

    def fit(self, parameters, config: Dict[str, Union[bool, bytes, float, int, str]])\
            -> Union[Tuple[any, any or float], int, Dict]:
        self.model.set_weights(parameters)

        batch_size: int = config['batch_size'] if 'batch_size' in config.keys(
        ) else 32
        epochs: int = config['local_epochs'] if 'local_epochs' in config.keys(
        ) else 1

        history = self.model.fit(
            x=self.x_train,
            y=self.y_train,
            epochs=epochs,
            # callbacks=[EarlyStopping(
            #     monitor='loss', patience=5, restore_best_weights=True)]
        )

        # self.model.save(f'models/model_{self.name}.h5')

        parameters_prime: list[ndarray] = self.model.get_weights()

        # print(textwrap.dedent(f'''
        #     ****************

        #     {parameters_prime}

        #     {history.history}

        #     ****************
        # '''))

        results = {
            'loss': history.history['loss'][0],
            # 'val_loss': history.history['val_loss'][0],
        }

        self.history['loss'].extend(history.history['loss'])
        # self.history['val_loss'].extend(history.history['val_loss'])
        self.history['weights'].append(
            [parameters_prime[0].tolist(), parameters_prime[1].tolist()])

        if config['final_round']:
            self.__save_history()

        # TODO see how to evaluate and save it

        return parameters_prime, len(self.x_train), results

    def evaluate(self, parameters, config: Dict[str, Union[bool, bytes, float, int, str]])\
            -> Union[tuple or any, int, Dict[str, float or any or tuple]]:
        self.model.set_weights(parameters)

        if not self.x_test or not self.y_test:
            raise Exception('Test variables are undefined or incomplete.')

        # Get config values
        if 'val_steps' in config.keys():
            steps: int = config['val_steps']
        else:
            steps = 5

        # Evaluate global model parameters on the local test data and return results
        loss = self.model.evaluate(
            self.x_test, self.y_test, len(self.x_test) // steps, steps=steps)

        num_examples_test = len(self.x_test)

        return loss, num_examples_test, {}

    def get_parameters(self):
        raise Exception(
            'Not implemented (server-side parameter initialization)')


if __name__ == '__main__':
    parser = ArgumentParser(description='Flower client.')
    parser.add_argument(
        '-c', '--client', default=0, help='the specific number client. if none is specified, the client will generate a new dataset to use.', type=int)
    args = parser.parse_args()

    with open('config.json', 'r') as f:
        config = json.load(f)

    if len(sys.argv) > 1:
        data = pd.read_csv(f'dataset/dataset{args.client}.csv')
        data = data.drop(data.columns[[0]], axis=1)
        data = data.drop(0)
        Y = data['Y']
        X = data.drop('Y', axis=1)
        print(X)
        print(Y)
    else:
        X, Y = create_dataset(100)

    # (x_train, x_test, y_train, y_test) = train_test_split(
    #     X.to_numpy(), Y.to_numpy(), train_size=0.8)

    client = TFclient(X.to_numpy(), Y.to_numpy(), num=args.client)
    fl.client.start_numpy_client('localhost:8080', client=client)
