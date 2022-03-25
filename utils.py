import os
import pickle
from typing import Dict, List

import pandas as pd
from scipy import stats
from tensorflow.python.keras import Model
from tensorflow.python.keras.models import load_model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def save_history(name: str, history: Dict) -> None:
    '''
    Saves a given history dictionary in the simulation directory
    by appending it to a list using pickle.

    Arguments:
      - name: name of the file
      - history: dictionary to save

    Returns none
    '''
    path: str = f'simulation/{name}'
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


def average_simulation() -> Dict:
    '''
    Averages all the simulations in the simulation directory.

    Returns a dictionary of the average values.
    '''
    directory: str = 'simulation'

    average: Dict = {"loss": [], "val_loss": []}

    for file in os.listdir(directory):
        with open(os.path.join(directory, file), 'rb') as data_file:
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
                    average[key][i] += value[i] / length

    return average


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

    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        model: Model = load_model(path)
        dict[file] = model.evaluate(x, y)

    return dict


def create_dataset(nb=10,
                   err_dist=stats.beta(a=8, b=2, scale=50/4),
                   means_dist=stats.norm(loc=0, scale=1),
                   draws_dist=stats.norm,
                   mean=90,
                   standard_deviation=2):
    """
    Creates a linear regression dataset based on a mean distribution and on an error distribution.
    X values are drawn from the player's distribution and Y is noisily drawn following:
        Y[j] ~ D[j](XT[j] teta[j] , epsilon**2[j]) where j is the player

    Arguments:
      - nb: number of data in the dataset
      - err_dist: distribution to draw error parameters from (err = epsilon**2) (scalar)
      - means_dist: distribution to draw X from
      - draws_dist: distribution to draw Y from: with mean*X as mean and variance epsilon^2

    Returns:
        Tuple of dataframes representing X and Y
    """

    X = pd.DataFrame(means_dist.rvs(nb))

    Y = pd.DataFrame(draws_dist(
        loc=X*mean,  # mean (μ)
        scale=standard_deviation  # standard deviation (σ)
    ).rvs())

    return (X, Y)
