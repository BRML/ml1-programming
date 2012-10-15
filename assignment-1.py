import warnings
import sklearn.neighbors.base
warnings.filterwarnings('ignore',
                        category=sklearn.neighbors.base.NeighborsWarning)

import matplotlib.pyplot as plt

from ml1assignments import (
    iris_data, knn, zero_one_loss, train_test_val_split,
    plot_decision_boundary)


if __name__ == '__main__':
    # Load data.
    X, z = iris_data('data/iris.data')
    X = X[:, (0, 2)]

    # Split data into training, validation and testing splits.
    RX, rz, VX, vz, TX, tz = train_test_val_split(X, z, 0.5, 0.25, 0.25)

    # Try out different values for k and find best k for the validation set.
    best_predict = None
    best_error = float('inf')
    for i in range(1, 10):
        predict = knn(RX, rz, i)

        predictions = predict(VX)
        error = zero_one_loss(vz, predictions)

        print 'Validation error using %i neighbours: %.4f' % (i, error)

        if error < best_error:
            best_predict = predict
            best_error = error

    x_extent = X[:, 0].min() - .2, X[:, 0].max() + .2
    y_extent = X[:, 1].min() - .2, X[:, 1].max() + .2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_decision_boundary(ax, best_predict, x_extent, y_extent)
    ax.scatter(TX[:, 0], TX[:, 1], c=tz, s=50)

    plt.show()
