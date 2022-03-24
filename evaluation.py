import os
import sys

import pandas as pd
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# tf.compat.v1.disable_eager_execution()


def evaluate_models(x, y):
    directory = 'models'

    dict = {}

    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        model = tf.keras.models.load_model(path)
        dict[file] = model.evaluate(x, y)

    print(dict)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        n = sys.argv[1]
        data = pd.read_csv(f'dataset/dataset{n}.csv')
    else:
        data = pd.read_csv('dataset/eval.csv')
    X = data['X']
    Y = data['Y']

    evaluate_models(X, Y)
