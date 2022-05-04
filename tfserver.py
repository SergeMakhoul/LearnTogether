import json
import os
from typing import Dict, Optional, Tuple

import flwr as fl
import pandas as pd
from flwr.common import Weights
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server
from flwr.server.strategy import FedAvg
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.initializers import initializers_v2
from tensorflow.python.keras.layers import Dense, InputLayer
from tensorflow.python.keras.optimizers import gradient_descent_v2

from utils import create_dataset, save_history

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_eval_fn(model: Model, server: Server = None):
    '''
    Return an evaluation function for server-side evaluation.
    '''

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    data = pd.read_csv(f'dataset/server.csv')
    data = data.drop(data.columns[[0]], axis=1)
    data = data.drop(0)
    Y = data['Y']
    X = data.drop('Y', axis=1)
    avr = {'loss': []}
    rnd = [0]

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)
        loss = model.evaluate(X, Y)

        avr['loss'].append(loss)

        rnd[0] += 1
        if rnd[0] == number_of_rounds:
            save_history('server', avr, 'simulation_server')

        return loss, {}

    return evaluate


def fit_config(rnd: int):
    '''
    Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    '''

    fit_config = {
        'batch_size': 32,
        'local_epochs': config['server']['fit_config']['local_epochs'],
        'final_round': True if rnd == number_of_rounds else False
    }

    return fit_config


def evaluate_config(rnd: int):
    '''Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    '''

    val_steps = {
        'val_steps': 1
    }

    return val_steps


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Load and compile model for
    #    1. server-side parameter initialization
    #    2. server-side parameter evaluation
    model = Sequential([
        InputLayer(input_shape=(2,)),
        Dense(units=1, kernel_initializer=initializers_v2.Zeros())
    ])
    model.compile(
        optimizer=gradient_descent_v2.SGD(
            learning_rate=config['model']['learning_rate']
        ),
        loss='mean_squared_error'
    )

    strat_config = config['server']['strategy']

    # Create strategy
    strategy = FedAvg(
        eval_fn=get_eval_fn(model),
        fraction_fit=strat_config['fraction_fit'],
        fraction_eval=strat_config['fraction_eval'],
        min_available_clients=strat_config['min_available_clients'],
        min_fit_clients=strat_config['min_fit_clients'],
        min_eval_clients=strat_config['min_eval_clients'],
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(
            model.get_weights()),
    )

    number_of_rounds = config['server']['number_of_rounds']

    # Start Flower server for four rounds of federated learning
    fl.server.start_server('localhost:8080',
                           config={'num_rounds': number_of_rounds},
                           strategy=strategy)
