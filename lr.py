import textwrap
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import learning_curve
import tensorflow as tf

from utils import create_dataset


class LinearRegression():
    def __init__(self, x_train, y_train, training_epochs=100, learning_rate=0.01):
        self.x_train = x_train
        self.y_train = y_train

        # Learning rate
        self.learning_rate = learning_rate

        # Number of loops for training through all your data to update the parameters
        self.training_epochs = training_epochs

        # Declare weights
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
    data = pd.read_csv(f'dataset/dataset0.csv')
    x = data['X']
    y = data['Y']

    # x, y = create_dataset(nb=10, mu=200, sigma=1)

    print(x, y, sep='\n\n')

    lr = LinearRegression(
        # x_train=[0.13948843576670042, 0.173527582180533, -0.06293171837371654, 0.6513469712066443, 0.6362507170987537, -
        #          0.2546161606992579, -0.7246301825533152, 0.37860505653108556, 0.8430679470238845, 1.7246587241209264],
        # y_train=[5.714483829232211, 3.1227738124799007, 17.536049787035733, -22.3359312608938, -12.908773996921543,
        #          2.2470554222055776, -27.975382585279878, 7.526552614556685, 6.673955685566586, 31.053467416864954],
        # x_train=[-0.14299087028749843, -0.9230061087901784, 1.0287143510396952, 0.6150624248275743, 0.7068517680142956, 2.0558309510152606, -1.3005132869354996, 0.9839073940254693, -1.2875802907325329, -1.5446761361995247, 1.2331599886872606, 0.5337614226042812, -0.15687804560932514, -0.18408987792584847, 0.11830930520864648, 1.6342176564082023, -0.2906085678621777, 0.2584444457899075, 0.10515887525151903, -1.2505769189286884, 0.09647796867032682, 1.1486387837044933, -0.8770359620887157, -0.058591555415109374, 0.748761929903176, -0.9182784807499089, 0.2740768159235774, -0.47252787228502113, 0.8754786004116838, 1.3892360557457033],
        # y_train=[5.107656791625155, 3.9757235464111584, 15.66441353249976, 6.711758363667992, -14.638633962874803, -1.2035152759835945, 14.403987056319501, 34.168361843299664, -8.618618104345545, -4.448869807054375, 21.54786848458467, -9.282614343081809, -6.913077590761094, -26.141773490244248, -6.907648806454278, 3.0229136797220697, 1.4744232880270522, 24.28416378971231, -0.7539642048399716, 3.410019705012827, -6.329027459736193, 5.0298217247372525, 7.83165581433571, -7.05423433342848, -2.968337684999436, -9.716067544911652, -11.499344185785734, -0.7277523977475495, 0.28321234725842825, 13.82603299730521],
        x_train=[[1.84918512, 1.41223274], [-0.70207466, -1.45338664], [-0.49924932, 0.76925332], [-0.15692759, 1.48629988], [-1.15467199, 0.42881861],
                 [0.5080076, 0.60105825], [-0.09968167, 0.29976003], [0.67102072, 0.58561806], [0.02674373, -1.45108705], [0.0262791, -1.56197261]],
        y_train=[1.0381180136809798, -1.3738763448693163, 5.174074463949539, 6.216879602149314, -5.482573183376867, - \
                 0.5089314625061117, 2.280481524703704, 3.8564209495643924, -2.264092807402001, -2.6708234735556924],
        training_epochs=1000,
        learning_rate=0.01
    )

lr.fit()
print(lr)
lr.plot()
