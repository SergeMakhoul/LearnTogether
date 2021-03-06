import os
import pickle
from typing import Dict, List

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from numpy import sqrt
from scipy import stats
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import load_model


def average_simulation(directory: str = 'simulation_history') -> Dict:
    '''
    Averages all the simulations in the simulation directory.

    Arguments:
      - directory: directory to search in for the simulation results

    Returns a dictionary of the average values.
    '''

    average: Dict = {
        'loss': [],
    }

    if not os.path.exists(directory):
        os.mkdir(directory)

    path = f'{directory}/{os.listdir(directory)[-1]}'

    d = [i for i in os.listdir(path) if not i.startswith('.')]

    for file in d:
        if file[0] == '.':
            continue

        with open(os.path.join(path, file), 'rb') as data_file:
            data = pickle.load(data_file)

        length = len(data)

        for iteration in data:
            for key, value in iteration.items():
                if key not in average.keys():
                    continue

                if len(average[key]) == 0:
                    for i in average.keys():
                        average[i] = [0 for _ in range(len(value))]

                for i in range(len(value)):
                    average[key][i] += value[i] / (length * len(d))

    return average


def average_one(client: str, directory: str = 'simulation_history') -> Dict:
    '''
    Averages all the simulations for one given player in the simulation directory.

    Arguments:
      - client: the client to average
      - directory: directory to search in for the simulation results

    Returns a dictionary of the average values.
    '''

    average: Dict = {
        'loss': [],
    }

    if not os.path.exists(directory):
        os.mkdir(directory)

    path = f'{directory}/{os.listdir(directory)[-1]}'

    with open(os.path.join(path, client), 'rb') as data_file:
        data = pickle.load(data_file)

    length = len(data)

    for iteration in data:
        for key, value in iteration.items():
            if key not in average.keys():
                continue

            if len(average[key]) == 0:
                for i in average.keys():
                    average[i] = [0 for _ in range(len(value))]

            for i in range(len(value)):
                average[key][i] += value[i] / (length)

    return average


def average_server():
    return average_one(client='server', directory='simulation_server_history')


def create_dataset(nb: int = 10, mu: int = 10, sigma: int = 1, mean: float or None = None):
    '''
    Creates a linear regression dataset based on a mean distribution and on an error distribution.
    X values are drawn from the player's distribution and Y is noisily drawn following:
        Y[j] ~ D[j](XT[j] teta[j] , epsilon**2[j]) where j is the player

    Arguments:
      - nb: number of data in the dataset
      - mu: mean of the label, ?? = E(??^2)
      - sigma: sigma squared is the variance of theta, ??^2 = var(??)

    Returns:
        Tuple of dataframes representing X and Y
    '''

    # means_dist_1 = stats.norm(loc=0, scale=1)
    # means_dist_2 = stats.norm(loc=3, scale=2)

    # params_dists = [means_dist_1, means_dist_2]

    means_dist = stats.norm(loc=0, scale=sigma)

    # means = pd.DataFrame([dist.rvs() for dist in params_dists]).T
    means = pd.DataFrame([mean if mean is not None else means_dist.rvs()]).T

    variance_dist = stats.beta(a=8, b=2, scale=(50/4)*(mu/10))

    # X = pd.DataFrame(stats.multivariate_normal(
    #     mean=np.array([0] * 2), cov=[[1.0, 0.0], [0.0, 1.0]]).rvs(nb))

    X = pd.DataFrame(stats.norm(loc=0, scale=1).rvs(nb))

    # Y = pd.DataFrame([stats.norm(
    #     X.dot(means.iloc[0])[j],
    #     # sqrt(variance_dist.rvs())
    #     sqrt(mu)
    # ).rvs() for j in range(nb)])

    Y = pd.DataFrame([stats.norm(
        X.dot(means).iloc[j],
        # sqrt(variance_dist.rvs())
        sqrt(mu)
    ).rvs() for j in range(nb)])

    return (X, Y, means[0][0])


def evaluate_models(x, y) -> Dict:
    '''
    Runs through all of the models saved in the models directory
    and evaluates them using the given x and y and saves the results
    in a dictionary for comparison.

    Arguments:
      - x: input data to compare
      - y: output data to compare

    Returns a dictionary with the name of the model as key and the evaluation as value
    '''
    directory = 'models'

    dict = {}

    if not os.path.exists(directory):
        raise 'Wrong Directory'

    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        model: Model = load_model(path)
        dict[file] = model.evaluate(x, y)

    return dict


def get_dataset(path: str):
    data = pd.read_csv(path)
    data.drop(data.columns[[0]], axis=1, inplace=True)
    data.drop(0, inplace=True)
    Y = data['Y']
    X = data['X']
    return (X, Y)


def save_history(name: str, history: Dict, directory: str) -> None:
    '''
    Saves a given history dictionary in the simulation directory
    by appending it to a list using pickle.

    Arguments:
      - name: name of the file
      - history: dictionary to save
      - directory: directory to save the file in

    Returns none
    '''
    if not os.path.exists(directory):
        os.mkdir(directory)

    path: str = f'{directory}/{name}'
    data: List[Dict] = []

    if not os.path.exists(path):
        with open(path, 'wb+') as f:
            data.append(history)
            pickle.dump(data, f)
        return

    with open(path, 'rb') as f:
        data = pickle.load(f)

    data.append(history)

    with open(path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    x, y, z = create_dataset(nb=100, mu=10)
    # print(x)
    # print(y)
    # print(z)
