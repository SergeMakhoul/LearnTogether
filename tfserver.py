import json
import os
from typing import Dict, Optional, Tuple

import flwr as fl
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, InputLayer
from tensorflow.python.keras.optimizers import gradient_descent_v2

from utils import create_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    X, Y = create_dataset(nb=10)

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)
        loss = model.evaluate(X, Y)
        return loss, {}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        # "local_epochs": 2 if rnd < 2 else 5,
        "local_epochs": 1,
        "final_round": True if rnd == number_of_rounds else False
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 1
    return {"val_steps": val_steps}


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = Sequential([
        InputLayer(input_shape=(1,)),
        Dense(1)
    ])
    model.compile(
        optimizer=gradient_descent_v2.SGD(
            learning_rate=config["model"]["learning_rate"]),
        loss='mean_squared_error')

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        # fraction_fit=0.3,
        # fraction_eval=0.2,
        # min_fit_clients=5,
        # min_eval_clients=3,
        # min_available_clients=5,
        # eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.weights_to_parameters(
            model.get_weights()),
    )

    number_of_rounds = config["server"]["number_of_rounds"]

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("localhost:8080",
                           config={"num_rounds": number_of_rounds},
                           strategy=strategy)
