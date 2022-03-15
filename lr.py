import textwrap
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


class LinearRegression():
    def __init__(self, datatsetPath='dataset/dataset.csv', training_epochs=100, learning_rate=0.01):
        self.datasetPath = datatsetPath
        self.x_train = pd.read_csv(self.datasetPath)['X'].to_numpy()
        self.y_train = pd.read_csv(self.datasetPath)['Y'].to_numpy()

        # Learning rate
        self.learning_rate = learning_rate

        # Number of loops for training through all your data to update the parameters
        self.training_epochs = training_epochs

        # declare weights
        self.weight = tf.Variable(0.)
        self.bias = tf.Variable(0.)

        # To store the training time
        self.training_time = 0

    def __linreg(self, x):
        y = self.weight * x + self.bias
        return y

    def __squared_error(self, y_pred, y_true):
        return tf.reduce_mean(tf.square(y_pred - y_true))

    def fit(self):
        t_start = perf_counter()
        for epoch in range(self.training_epochs):
            # Compute loss within Gradient Tape context
            with tf.GradientTape() as tape:
                y_predicted = self.__linreg(self.x_train)
                loss = self.__squared_error(y_predicted, self.y_train)

                # Get gradients
                gradients = tape.gradient(loss, [self.weight, self.bias])

                # Adjust weights
                self.weight.assign_sub(gradients[0]*self.learning_rate)
                self.bias.assign_sub(gradients[1]*self.learning_rate)

                # Print output
                print(f"Epoch count {epoch}: Loss value: {loss.numpy()}")

        self.training_time = perf_counter() - t_start

    def get_weights(self):
        return (self.weight.numpy(), self.bias.numpy())

    def get_training_time(self):
        return self.training_time

    def plot(self):
        plt.scatter(self.x_train, self.y_train)
        plt.plot(self.x_train, self.__linreg(self.x_train), 'r')
        plt.show()

    def __str__(self):
        return(textwrap.dedent(f"""
        **********

        Weights of the model: {self.get_weights()}

        Training time: {self.get_training_time()}

        **********
        """))


if __name__ == '__main__':
    lr = LinearRegression()
    lr.fit()
    print(lr)
    # lr.plot()
