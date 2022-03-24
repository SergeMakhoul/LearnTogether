import os
import pickle
from typing import Dict, List

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import Model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def save_history(name: str, history: Dict):
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
    directory = 'models'

    dict = {}

    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        model: Model = load_model(path)
        dict[file] = model.evaluate(x, y)

    return dict
