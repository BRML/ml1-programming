# -*- coding utf-8 -*-

__author__ = "Justin Bayer, bayer.justin@googlemail.com"

import math
import random

import numpy as np
import pylab


def iris_data(fn):
    with open(fn) as fp:
        lines = fp.readlines()
    # Remove whitespace.
    lines = [i.strip() for i in lines]
    # Remove empty lines.
    lines = [i for i in lines if i]
    # Split by comma.
    lines = [i.split(',') for i in lines]
    # Inputs are the first four elements.
    inpts = [i[:4] for i in lines]
    # Labels are the last.
    labels = [i[-1] for i in lines]

    # Make arrays out of the inputs, one row per sample.
    X = np.empty((150, 4))
    X[:] = inpts

    # Make integers array out of label strings.
    #
    # We do this by first creating a set out of all labels to remove
    # any duplicates. Then we create a dictionary which maps label
    # names to an index. Afterwards, we loop over all labels and
    # assign the corresponding integer to that field in the label array z.
    z = np.empty(150)
    label_names = sorted(set(labels))
    label_to_idx = dict((j, i) for i, j in enumerate(label_names))

    for i, label in enumerate(labels):
        z[i] = label_to_idx[label]

    return X, z


def knn(X, z, k):
    """Return a function to do k nearest neighbour prediction.

    The function returned will do a majority vote among the k nearest
    neighbours.

    :param X: An (n, d) sized array holding n data items of dimensionality d.
    :param z: An n sized vector holding integers that indicate the class of the
        corresponding item in X. Integers start at 0 and end at c-1, where c is
        the number of classes.
    :param k: Number of neighbours to use.
    """
    def predict(x):
        # TODO: Calculate the distance of x to every point in the training set
        # X.

        # TODO: Pick the k points with the lowest distance.

        # TODO: Do a majority vote and return the class as an integer.

    return predict


def train_test_val_split(X, Z, train_frac, val_frac, test_frac):
    """Split the data into three sub data sets, one for training, one for
    validation and one for testing. The data is shuffled first."""
    assert train_frac + val_frac + test_frac == 1, "fractions don't sum up to 1"

    n_samples = X.shape[0]
    n_samples_train = int(math.floor(n_samples * train_frac))
    n_samples_val = int(math.floor(n_samples * val_frac))

    idxs = range(n_samples)
    random.shuffle(idxs)
    train_idxs = idxs[:n_samples_train]
    val_idxs = idxs[n_samples_train:n_samples_train + n_samples_val]
    test_idxs = idxs[n_samples_train + n_samples_val:]

    return (X[train_idxs], Z[train_idxs],
            X[val_idxs], Z[val_idxs],
            X[test_idxs], Z[test_idxs])


def plot_decision_boundary(ax, predict, x_extent, y_extent):
    """Plot the decision boundary of a classification decision function
    `predict` to axis ax. The pairs `x_extent` and `y_extent` give the
    (min, max) values of the plot."""
    h = 0.04
    x_min, x_max = x_extent
    y_min, y_max = y_extent

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.pcolormesh(xx, yy, Z, alpha=.5)
