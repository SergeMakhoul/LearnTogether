import flwr as fl
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from create_dataset import create_dataset
from typing import Dict


def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model: LinearRegression):
    """Return an evaluation function for server-side evaluation."""

    # df = pd.read_csv('dataset.csv')

    # Y = df['Y'].to_numpy()
    # Y = Y.astype('int')

    # X = df['X'].to_numpy()
    # X = X.reshape(-1, 1)

    X, Y = create_dataset()

    (x_train, x_test, y_train, y_test) = train_test_split(
        X.to_numpy(), Y.to_numpy(), train_size=0.75)

    (x_train, x_test, y_train, y_test) = train_test_split(X, Y, train_size=0.75)

   # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        model.coef_ = parameters[0]
        if model.fit_intercept:
            model.intercept_ = parameters[1]

        loss = mean_squared_error(y_test, model.predict(x_test))
        accuracy = model.score(x_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    model = LinearRegression()

    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )

    fl.server.start_server(
        'localhost:8080', strategy=strategy, config={'num_rounds': 5})
