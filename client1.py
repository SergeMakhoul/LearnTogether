import flwr as fl
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from dataset.create_dataset import create_dataset
from model import Linear

# df = pd.read_csv('dataset.csv')

# Y = df.loc[:5, 'Y'].to_numpy()
# Y = Y.astype('int')

# X = df.loc[:5, 'X'].to_numpy()
# X = X.reshape(-1, 1)

X, Y = create_dataset()

(x_train, x_test, y_train, y_test) = train_test_split(
    X.to_numpy(), Y.to_numpy(), train_size=0.75)

client = Linear(x_train, y_train, x_test, y_test)

fl.client.start_numpy_client('localhost:8080', client=client)
