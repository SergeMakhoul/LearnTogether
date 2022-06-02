import json
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from utils import create_dataset

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Create a dataset and save it into csv files.')
    parser.add_argument('-s', '--seed', default=1234,
                        help='seed for data generation', type=int)
    parser.add_argument('-c', '--clients', default=10,
                        help='number of clients to generate data for', type=int)
    args = parser.parse_args()

    with open('config.json', 'r') as input:
        config = json.load(input)

    data_config = config['data']

    np.random.seed(seed=args.seed)

    # Creating the data for the server
    # We create the server's dataset before the clients to ensure
    # that the server will have the same dataset every simulation
    # and not be related to the number of clients
    print('[INFO] Dataset | Creating server dataset')

    X, Y = create_dataset(
        nb=1000,
        mu=data_config['mu']
    )

    DATASET_PATH = f'dataset/seed_{args.seed}'

    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    dataset = pd.concat([X, Y], axis=1, keys=['X', 'Y'])
    dataset.to_csv(f'{DATASET_PATH}/server.csv')

    # Creating the data for the clients
    print('[INFO] Dataset | Creating client datasets')
    for i in range(args.clients):
        X, Y = create_dataset(
            nb=data_config['number_of_samples'],
            mu=data_config['mu']
        )

        dataset = pd.concat([X, Y], axis=1, keys=['X', 'Y'])
        dataset.to_csv(f'{DATASET_PATH}/dataset{i}.csv')
