import json
import sys

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

    # creating the data for the clients
    for i in range(n):
        X, Y = create_dataset(
            nb=data_config['number_of_samples'],
            mu=data_config['mu']
        )

        dataset = pd.concat([X, Y], axis=1, keys=['X', 'Y'])
        dataset = dataset.sample(frac=1)
        dataset.to_csv(f'dataset/dataset{i}.csv')

    # creating the data for the server
    X, Y = create_dataset(
        nb=1000,
        mu=data_config['mu']
    )

    dataset = pd.concat([X, Y], axis=1, keys=['X', 'Y'])
    dataset = dataset.sample(frac=1)
    dataset.to_csv(f'dataset/server.csv')
