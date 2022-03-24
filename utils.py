import os
import pickle
from typing import Dict, List

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import Model

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


def evaluate_models(x, y) -> dict:
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
