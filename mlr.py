from typing import Tuple

import pandas as pd

from utils import create_dataset


def calculate_parameters(x1: pd.DataFrame, x2: pd.DataFrame, y: pd.DataFrame) -> Tuple[float, float, float]:
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
    avr = []
    nb = 10
    mu = 200
    for _ in range(1):
        data = pd.read_csv(f'dataset/dataset0.csv')
        data = data.drop(data.columns[[0]], axis=1)
        data = data.drop(0)
        y = data['Y']
        x = data.drop('Y', axis=1)
        # x, y = create_dataset(nb, mu)
        # x = pd.DataFrame([
        #     [-0.737513, 1.158807],
        #     [1.709208, -0.977309],
        #     [-0.363734, 0.031622],
        #     [-1.906486, -0.730987],
        #     [-0.191485, 0.686581],
        #     [0.517718, -0.757618],
        #     [0.224554, -0.413411],
        #     [2.416377, -0.583175],
        #     [0.881766, -1.392606],
        #     [-0.400346, -0.308192]
        # ])

        # y = pd.DataFrame([16.982345912775642, -5.605549871297961, 12.280438110508962, 13.539815722501146, -6.650178714917922,
        #                  -7.023483784834454, 7.714626007844384, -3.878709595474202, 33.74824705433084, 15.713715047604907])

        # print(x['X'])

        x1 = pd.DataFrame(x['X'].values)
        x2 = pd.DataFrame(x['X.1'].values)

        (a, b1, b2) = calculate_parameters(x1, x2, y)

        x_test, y_test = create_dataset(nb, mu)

        x1_test = pd.DataFrame(x_test[0].values)
        x2_test = pd.DataFrame(x_test[1].values)

        # y_pred = x1_test * b1 + x2_test * b2 + a
        y_pred = x1 * b1 + x2 * b2 + a

        mse = ((((y_pred - y) ** 2).sum()) / y.size)
        avr.append(mse)

        print(f'y = {b1} * X1 + {b2} * X2 + {a}')
        print(mse)

    print(
        f'Average MSE with parameters mu={mu} and nb={nb} was: {avr}')
