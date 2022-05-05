import json
import sys

import numpy as np
import pandas as pd

from utils import create_dataset

if __name__ == '__main__':
    with open('config.json', 'r') as input:
        config = json.load(input)

    data_config = config['data']

    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 10

    np.random.seed(seed=1234)

    # Creating the data for the server
    # We create the server's dataset before the clients to ensure
    # that the server will have the same dataset every simulation
    # and not be related to the number of clients
    print('[INFO] Dataset | Creating server dataset')

    X, Y = create_dataset(
        nb=1000,
        mu=data_config['mu']
    )

    dataset = pd.concat([X, Y], axis=1, keys=['X', 'Y'])
    dataset.to_csv(f'dataset/server.csv')

    # Creating the data for the clients
    print('[INFO] Dataset | Creating client datasets')
    for i in range(n):
        X, Y = create_dataset(
            nb=data_config['number_of_samples'],
            mu=data_config['mu']
        )

        dataset = pd.concat([X, Y], axis=1, keys=['X', 'Y'])
        dataset.to_csv(f'dataset/dataset{i}.csv')
