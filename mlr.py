import json
import os
from argparse import ArgumentParser
from typing import List, Tuple

from numpy import mean
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


def one_seed(seed_file: str or int, noc: int = None):
    if type(seed_file) == int:
        seed_file = f'seed_{seed_file}'

    num_of_clients = noc if noc is not None else args.num_of_clients

    weights = {'a': [], 'b': []}

    for i in range(num_of_clients):
        x, y = get_dataset(f'dataset/{seed_file}/client_{i}/dataset.csv')

        a, b = linear_regression(x, y)

        weights['a'].append(a)
        weights['b'].append(b)

    a = mean(weights['a'])
    b = mean(weights['b'])

    mse_list = []

    for i in range(num_of_clients):
        x_test, y_test = get_dataset(
            f'dataset/{seed_file}/client_{i}/test.csv')

        y_pred = x_test * b + a

        mse = mean((y_pred - y_test) ** 2)
        mse_list.append({'client': i, 'mse': mse})

    mse_avr = mean([i['mse'] for i in mse_list])

    return {
        'num_of_clients': num_of_clients,
        'mse_avr': mse_avr,
        'mse_list': mse_list,
        'a': a,
        'b': b
    }, weights


def all_seeds(data: json or List, mse_avr: json or List, noc: int = None):
    dir = os.listdir('dataset')
    dir.sort()

    num_of_clients = noc if noc is not None else args.num_of_clients

    for seed_file in dir:
        if seed_file.startswith('.'):
            continue

        sim_data, sim_weights = one_seed(
            seed_file=seed_file, noc=num_of_clients)

        seed = int(seed_file.split('_')[1])

        index = next(
            (i for i, item in enumerate(data) if item['seed'] == seed),
            None)

        if index == None:
            data.append({
                'seed': seed,
                'data': [sim_data],
                'weights': sim_weights
            })
            mse_avr.append({
                'seed': seed,
                'data': [{
                    'num_of_clients': num_of_clients,
                    'mse_avr': sim_data['mse_avr']
                }]
            })
        else:
            data[index]['data'].append(sim_data)
            data[index]['weights'] = sim_weights
            mse_avr[index]['data'].append({
                'num_of_clients': num_of_clients,
                'mse_avr': sim_data['mse_avr']
            })

    return data, mse_avr


def open_json(path: str) -> json or List:
    if not os.path.exists(path):
        open(path, 'a').close()
        return []

    with open(path, 'r') as input_file:
        try:
            res = json.load(input_file)
        except json.decoder.JSONDecodeError:
            res = []

    return res


if __name__ == '__main__':
    parser = ArgumentParser(description='Flower client.')
    parser.add_argument(
        '-s',
        '--seed',
        default=None,
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
    # parser.add_argument(
    #     '-c',
    #     '--continue',
    #     action='store_true',
    #     help='whether to create a new JSON file or not'
    # )
    args = parser.parse_args()

    data = open_json('out.json')
    mse = open_json('mse.json')

    # if args.seed is None:
    #     res, mse_avr = all_seeds(data, mse, i)
    # else:
    #     sim_data, sim_weights = one_seed(args.seed)
    #     res = {
    #         'seed': args.seed,
    #         'data': [sim_data],
    #         'weights': sim_weights
    #     }
    #     mse_avr = {
    #         'seed': args.seed,
    #         'mse_avr': sim_data['mse_avr']
    #     }

    for i in range(1, args.num_of_clients):
        print(i)
        res, mse_avr = all_seeds(data=data, mse_avr=mse, noc=i)

        data = res
        mse = mse_avr

    with open('out.json', 'w+') as output:
        json.dump(data, output)

    with open('mse.json', 'w+') as output:
        json.dump(mse, output)
