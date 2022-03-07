from typing import Tuple
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

# means_dist_1 = stats.norm(loc=0, scale=1)
# variance_dist = stats.beta(a=8, b=2, scale=50/4)


def create_dataset(nb=5,
                   err_dist=stats.beta(a=8, b=2, scale=50/4),
                   means_dist=stats.norm(loc=0, scale=1),
                   draws_dist=stats.norm):
    """
    Creates a linear regression dataset based on a mean distribution and on an error distribution.
    X values are drawn from the player's distribution and Y is noisily drawn following:
        Y[j] ~ D[j](XT[j] teta[j] , epsilon**2[j]) where j is the player

    Arguments:
      - nb: number of data in the dataset
      - err_dist: distribution to draw error parameters from (err = epsilon**2) (scalar)
      - means_dist: distribution to draw X from
      - draws_dist: distribution to draw Y from: with mean*X as mean and variance epsilon^2

    Returns:
        Tuple of pandas dataframes representing X and Y
    """

    # x_cov = [[1.0, 0.0], [0.0, 1.0]]
    # D = len(x_cov)
    # mean_x = np.array([0] * D)
    # x_dist = stats.multivariate_normal(mean=mean_x, cov=x_cov)

    # means = pd.DataFrame([dist.rvs() for dist in params_dists]).T
    # X = pd.DataFrame(x_dist.rvs(nb))

    means = pd.DataFrame([means_dist.rvs()])
    X = pd.DataFrame(means_dist.rvs(nb))

    # The location (loc) keyword specifies the mean.
    # The scale (scale) keyword specifies the standard deviation.
    Y = pd.DataFrame(draws_dist(
        loc=X.dot(means),  # mean of this player
        scale=np.sqrt(err_dist.rvs())  # error of this player
    ).rvs())

    # print(f'X: length = {len(X)}\n{X}\n')
    # print(f'Y: length = {len(Y)}\n{Y}\n')

    # plt.scatter(X, Y)
    # plt.show()

    return (X, Y)


# create_dataset()
