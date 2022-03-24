import os
import pickle
from typing import Dict, List


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
