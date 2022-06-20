import json
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from utils import create_dataset


def create_v1():
    theta_dic = {}

    X, Y, mean = create_dataset(
        nb=1000,
        mu=data_config['mu'],
        sigma=data_config['sigma']
    )

    theta_dic['server'] = mean

    DATASET_PATH = f'dataset/seed_{args.seed}'

    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    dataset = pd.concat([X, Y], axis=1, keys=['X', 'Y'])
    dataset.to_csv(f'{DATASET_PATH}/server.csv')

    theta_dic['clients'] = []

    for i in range(args.clients):
        X, Y, mean = create_dataset(
            nb=data_config['number_of_samples'],
            mu=data_config['mu'],
            sigma=data_config['sigma']
        )

        theta_dic['clients'].append({'client': i, 'theta': mean})

        dataset = pd.concat([X, Y], axis=1, keys=['X', 'Y'])
        dataset.to_csv(f'{DATASET_PATH}/dataset{i}.csv')

    with open(f'{DATASET_PATH}/conf.json', 'w+') as f:
        json.dump({
            'nb': data_config['number_of_samples'],
            'mu': data_config['mu'],
            'sigma': data_config['sigma'],
            'theta': theta_dic
        }, f)


def create_v2():
    DATASET_PATH = f'dataset/seed_{args.seed}'

    theta_dic = []

    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    for i in range(args.clients):
        os.mkdir(f'{DATASET_PATH}/client_{i}')

        x_data, y_data, mean = create_dataset(
            nb=data_config['number_of_samples'],
            mu=data_config['mu'],
            sigma=data_config['sigma']
        )

        dataset = pd.concat([x_data, y_data], axis=1, keys=['X', 'Y'])
        dataset.to_csv(f'{DATASET_PATH}/client_{i}/dataset.csv')

        x_test, y_test, _ = create_dataset(
            nb=1000,
            mu=data_config['mu'],
            sigma=data_config['sigma'],
            mean=mean
        )

        theta_dic.append({'client': i, 'theta': mean})

        dataset = pd.concat([x_test, y_test], axis=1, keys=['X', 'Y'])
        dataset.to_csv(f'{DATASET_PATH}/client_{i}/test.csv')

    with open(f'{DATASET_PATH}/conf.json', 'w+') as f:
        json.dump({
            'nb': data_config['number_of_samples'],
            'mu': data_config['mu'],
            'sigma': data_config['sigma'],
            'theta': theta_dic
        }, f)


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

    create_v2()
