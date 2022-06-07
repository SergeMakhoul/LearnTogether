import json
from argparse import ArgumentParser
from typing import Tuple

from pandas import DataFrame as df

from utils import create_dataset, get_dataset


def linear_regression(x: df, y: df) -> Tuple[float, float]:
    mx = x.mean()
    my = y.mean()

    ssx = ((x - mx)**2).sum()
    sp = ((x - mx) * (y - my)).sum()

    b = sp / ssx
    a = my - b * mx

    return (a, b)


def multiple_regression(x1: df, x2: df, y: df) -> Tuple[float, float, float]:
    mx1 = x1.mean()[0]
    mx2 = x2.mean()[0]
    my = y.mean()

    sx1 = x1.sum()[0]
    sx2 = x2.sum()[0]
    sy = y.sum()

    s = x1.size

    ssx1 = ((x1 - mx1)**2).sum()[0]
    ssx2 = ((x2 - mx2)**2).sum()[0]

    sx1y = (x1 * y).sum()[0]
    spx1y = sx1y - sx1 * sy / s

    sx2y = (x2 * y).sum()[0]
    spx2y = sx2y - sx2 * sy / s

    sx1x2 = (x1 * x2).sum()[0]
    spx1x2 = sx1x2 - sx1 * sx2 / s

    b1 = (spx1y * ssx2 - spx1x2 * spx2y) / (ssx1 * ssx2 - spx1x2 * spx1x2)
    b2 = (spx2y * ssx1 - spx1x2 * spx1y) / (ssx1 * ssx2 - spx1x2 * spx1x2)
    a = my - b1 * mx1 - b2 * mx2

    return (a, b1, b2)


if __name__ == '__main__':
    parser = ArgumentParser(description='Flower client.')
    parser.add_argument(
        '-s',
        '--seed',
        default=1234,
        help='the seed for the dataset',
        type=int
    )
    parser.add_argument(
        '-n',
        '--num_of_clients',
        default=1,
        help='total number of clients to simulate',
        type=int
    )
    args = parser.parse_args()

    with open('config.json', 'r') as f:
        config = json.load(f)

    weights = {'a': 0, 'b': 0}

    for i in range(args.num_of_clients):
        x, y = get_dataset(f'dataset/seed_{args.seed}/dataset{i}.csv')

        a, b = linear_regression(x, y)

        weights['a'] += a
        weights['b'] += b

    weights['a'] /= args.num_of_clients
    weights['b'] /= args.num_of_clients

    a = weights['a']
    b = weights['b']

    x_test, y_test = get_dataset(f'dataset/seed_{args.seed}/server.csv')

    y_pred = x_test * b + a

    mse = ((((y_pred - y_test) ** 2).sum()) / y_test.size)

    print(f'seed = {args.seed}')
    print(f'num_of_clients = {args.num_of_clients}')
    print(f'y = {b} * X + {a}')
    print(f'mse = {mse}')
    print('###############')

    # mu = config['data']['mu']
    # nb = config['data']['number_of_samples']

    # print(f'Average MSE with parameters mu={mu} and nb={nb} was: {mse}')
