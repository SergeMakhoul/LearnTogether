import json
import os
from argparse import ArgumentParser
from typing import Dict, Optional, Tuple

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import flwr as fl
import numpy as np
import pandas as pd
from flwr.common import Weights
from flwr.server.server import Server
from flwr.server.strategy import FedAvg
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.initializers import initializers_v2
from tensorflow.python.keras.layers import Dense, InputLayer
from tensorflow.python.keras.optimizers import gradient_descent_v2

from utils import get_dataset, save_history


# def get_eval_fn(model: Model, server: Server = None):
#     '''
#     Return an evaluation function for server-side evaluation.
#     '''

#     # Load data and model here to avoid the overhead of doing it in `evaluate` itself
#     X, Y = get_dataset(f'dataset/seed_{args.seed}/server.csv')
#     avr = {'loss': []}
#     rnd = [0]

#     # The `evaluate` function will be called after every round
#     def evaluate(
#         weights: Weights,
#     ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
#         model.set_weights(weights)
#         loss = model.evaluate(X, Y)
#         # loss = model.evaluate(X, Y)\
#         #     + ((config['data']['mu'] / (config['data']['number_of_samples'] - 2)
#         #         - (config['data']['mu'] / (config['data']['number_of_samples'] -
#         #            2) - config['data']['sigma'] ** 2) / config['cost_function']['T']
#         #         - config['data']['sigma'] ** 2) / np.exp(2 * config['cost_function']['lambda'] * config['cost_function']['s'] * config['cost_function']['T']))\
#         #     * np.exp(2 * config['cost_function']['lambda'] * config['cost_function']['s'] * strat_config['min_available_clients'])

#         # loss = model.evaluate(
#         #     X, Y) + (config['cost'] * (strat_config['min_available_clients'] - 1))

#         avr['loss'].append(loss)

#         rnd[0] += 1
#         if rnd[0] == number_of_rounds:
#             save_history(
#                 name='server',
#                 history=avr,
#                 directory=args.dir
#             )

#         return loss, {}

#     return evaluate


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
    '''
    Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    '''

    val_steps = {
        'val_steps': 10,
        'final_round': True if rnd == number_of_rounds else False
    }

    return val_steps


if __name__ == '__main__':
    parser = ArgumentParser(description='Flower server.')
    parser.add_argument(
        '-c',
        '--clients',
        help='the number of clients this server expects',
        type=int
    )
    parser.add_argument(
        '-s',
        '--seed',
        help='the seed for this server\'s dataset',
        type=int
    )
    parser.add_argument(
        '-d',
        '--dir',
        help='directory for the history',
        type=str
    )
    parser.add_argument(
        '-p',
        '--port',
        default=8080,
        help='port to use for connection.',
        type=int
    )
    args = parser.parse_args()

    with open('config.json', 'r') as f:
        config = json.load(f)

    # Load and compile model for
    #    1. server-side parameter initialization
    #    2. server-side parameter evaluation
    model = Sequential([
        InputLayer(input_shape=(1,)),
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
        # eval_fn=get_eval_fn(model),
        fraction_fit=strat_config['fraction_fit'],
        fraction_eval=strat_config['fraction_eval'],
        min_available_clients=args.clients,
        min_fit_clients=args.clients,
        min_eval_clients=args.clients,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(
            model.get_weights()),
    )

    number_of_rounds = config['server']['number_of_rounds']

    no_err = False
    nb_tries = 0
    while not no_err:
        try:
            fl.server.start_server(
                f'localhost:{args.port}',
                config={'num_rounds': number_of_rounds},
                strategy=strategy
            )
            no_err = True
        except:
            # nb_tries += 1
            continue
