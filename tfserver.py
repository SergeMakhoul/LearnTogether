from typing import Dict, Optional, Tuple

import flwr as fl
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.optimizers import gradient_descent_v2
from tensorflow.python.keras.layers import Dense, InputLayer

from dataset.create_dataset import create_dataset


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    X, Y = create_dataset(nb=20)
    (x_train, x_test, y_train, y_test) = train_test_split(
        X.to_numpy(), Y.to_numpy(), train_size=0.75)

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)
        loss = model.evaluate(x_test, y_test)
        return loss, {}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
        # "local_epochs": 10,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = Sequential([
        InputLayer(input_shape=(1,)),
        Dense(1)
    ])
    model.compile(
        optimizer=gradient_descent_v2.SGD(learning_rate=0.01),
        loss='mean_squared_error')

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_eval=0.2,
        min_fit_clients=3,
        min_eval_clients=3,
        min_available_clients=10,
        # eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(
            model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("localhost:8080",
                           config={"num_rounds": 7},
                           strategy=strategy)
