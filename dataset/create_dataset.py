import numpy as np
import pandas as pd
from scipy import stats


def create_dataset(nb=5,
                   err_dist=stats.beta(a=8, b=2, scale=50/4),
                   means_dist=stats.norm(loc=0, scale=1),
                   draws_dist=stats.norm,):
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
        Tuple of dataframes representing X and Y
    """

    means = pd.DataFrame([means_dist.rvs()])
    X = pd.DataFrame(means_dist.rvs(nb))

    # The location (loc) keyword specifies the mean.
    # The scale (scale) keyword specifies the standard deviation.
    # Y = pd.DataFrame(draws_dist(
    #     loc=X.dot(means),  # mean of this player
    #     scale=np.sqrt(err_dist.rvs())  # error of this player
    # ).rvs())

    Y = pd.DataFrame(draws_dist(
        loc=X*90,  # mean of this player
        scale=4  # error of this player
    ).rvs())

    return (X, Y)


# create_dataset()
