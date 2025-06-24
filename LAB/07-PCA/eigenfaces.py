"""
Eigenfaces main script.
"""

import numpy as np
from utils import show_eigenfaces
import matplotlib.pyplot as plt

plt.ion()


class Eigenfaces:
    """
    Performs PCA to project faces in a reduced space.
    """

    def __init__(self, n_components: int):
        """
        Parameters
        ----------
        n_components: int
            number of principal component
        """
        self.n_components = n_components

        # Per-feature empirical mean, estimated from the training set.
        self._x_mean = None

        # Principal axes in feature space, representing the directions
        # of maximum variance in the data.
        self._ef = None

    def fit(self, X: np.ndarray, verbose: bool = False):
        """
        Parameters
        ----------
        X: ndarray
            Training set will be used for fitting the PCA
                (shape: (n_samples, w*h))
        verbose: bool
        """

        # compute mean vector and store it for the inference stage
        self._x_mean = np.mean(X, axis=0)

        if verbose:
            # show mean face
            plt.imshow(np.reshape(self._x_mean, newshape=(112, 92)), cmap="gray")
            plt.title("mean face")
            plt.waitforbuttonpress()
            plt.close()

        # remove mean from the data
        X_norm = X - self._x_mean

        # compute covariance with the eigen-trick
        cov = X_norm @ X_norm.T

        # compute eigenvectors of the covariance matrix
        eigval, eigvec = np.linalg.eig(cov)

        # sort them (decreasing order w.r.t. eigenvalues)
        idx = np.argsort(eigval)[::-1]
        sorted_eigval, sorted_eigvec = eigval[idx], eigvec[:, idx]

        # find best n_components (iperparam)
        """ sorted_eigval_norm = sorted_eigval / np.sum(sorted_eigval)
        somma = np.cumsum(sorted_eigval_norm)
        self.n_components = np.argmax(somma >= 0.95)
        print(f"Number of components to retain 95% of variance: {self.n_components}") """

        # select principal components (vectors)
        principal_sorted_eigval, principal_sorted_eigvec = (
            sorted_eigval[: self.n_components],
            sorted_eigvec[:, : self.n_components],
        )

        # retrieve original eigenvec
        self._ef = X.T @ principal_sorted_eigvec

        # normalize the retrieved eigenvectors to have unit length
        self._ef /= np.sqrt(principal_sorted_eigval)

        if verbose:
            # show eigenfaces
            show_eigenfaces(self._ef, (112, 92))
            plt.waitforbuttonpress()
            plt.close()

    def transform(self, X: np.ndarray):
        """
        Parameters
        ----------
        X: ndarray
            faces to project (shape: (n_samples, w*h))
        Returns
        -------
        out: ndarray
            projections (shape: (n_samples, self.n_components)
        """

        # project faces according to the computed directions
        return X @ self._ef

    def inverse_transform(self, X: np.ndarray):
        return X @ self._ef.T


class NearestNeighbor:

    def __init__(self):
        self._X_db, self._Y_db = None, None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        Fit the model using X as training data and y as target values
        """
        self._X_db = X
        self._Y_db = y

    def predict(self, X: np.ndarray):
        """
        Finds the 1-neighbor of a point. Returns predictions as well as indices of
        the neighbors of each point.
        """
        num_test_samples = X.shape[0]

        # predict test faces
        predictions = np.zeros((num_test_samples,))
        nearest_neighbors = np.zeros((num_test_samples,), dtype=np.int32)

        for i in range(num_test_samples):

            distances = np.sum(np.square(self._X_db - X[i]), axis=1)

            # nearest neighbor classification
            nearest_neighbor = np.argmin(distances)
            nearest_neighbors[i] = nearest_neighbor
            predictions[i] = self._Y_db[nearest_neighbor]

        return predictions, nearest_neighbors
