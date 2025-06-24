"""
Eigenfaces main script.
"""

import numpy as np
from utils import show_nearest_neighbor
from data_io import get_faces_dataset

from eigenfaces import Eigenfaces, NearestNeighbor
import matplotlib.pyplot as plt

plt.ion()


def main(iperparam=200):

    # get_data
    X_train, Y_train, X_test, Y_test = get_faces_dataset(
        path="att_faces", train_split=0.6
    )

    # number of principal components
    n_components = iperparam

    # fit the PCA transform
    eigpca = Eigenfaces(n_components)
    eigpca.fit(X_train, verbose=True)

    # project the training data
    proj_train = eigpca.transform(X_train)

    # project the test data
    proj_test = eigpca.transform(X_test)

    # fit a 1-NN classifier on PCA features
    nn = NearestNeighbor()
    nn.fit(proj_train, Y_train)

    # Compute predictions and indices of 1-NN samples for the test set
    predictions, nearest_neighbors = nn.predict(proj_test)

    # Compute the accuracy on the test set
    test_set_accuracy = float(np.sum(predictions == Y_test)) / len(predictions)
    print(f"Test set accuracy: {test_set_accuracy} - Iperparam = {iperparam}")

    # Show results.
    show_nearest_neighbor(X_train, Y_train,
                         X_test, Y_test, nearest_neighbors)
    return test_set_accuracy

# comment the plot methods and call the fit with verbose=False
def brute_force_best_iperparam():
    best_iperparam = 0
    best_accuracy = 0
    for iperparam in range(1000):
        cur = main(iperparam)
        if cur > best_accuracy:
            best_accuracy = cur
            best_iperparam = iperparam

    print(f"Best iperparam is {best_iperparam} with {best_accuracy} accuracy!!")


if __name__ == "__main__":
    main()
