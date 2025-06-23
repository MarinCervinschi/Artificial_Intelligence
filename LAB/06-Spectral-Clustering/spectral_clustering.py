import numpy as np
from datasets import two_moon_dataset, gaussians_dataset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.ion()


def spectral_clustering(data, n_cl, sigma=1., fiedler_solution=False):
    """
    Spectral clustering.

    Parameters
    ----------
    data: ndarray
        data to partition, has shape (n_samples, dimensionality).
    n_cl: int
        number of clusters.
    sigma: float
        std of radial basis function kernel.
    fiedler_solution: bool
        return fiedler solution instead of kmeans

    Returns
    -------
    ndarray
        computed assignment. Has shape (n_samples,)
    """
    # compute affinity matrix

    #affinity_matrix = np.exp(-np.sum(np.square(data[:, None] - data[None, :]), axis=-1)/sigma**2)
    affinity_matrix = np.exp(-np.linalg.norm(data[np.newaxis, :] - data[:, np.newaxis], axis=2)/sigma**2)

    # compute degree matrix
    # degree_matrix = np.eye(affinity_matrix.shape) @ np.sum(affinity_matrix, axis=1)
    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))

    # compute laplacian
    # laplacian_matrix = np.diff(degree_matrix, affinity_matrix)
    laplacian_matrix = degree_matrix - affinity_matrix

    # compute eigenvalues and vectors (suggestion: np.linalg is your friend)
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)

    # ensure we are not using complex numbers - you shouldn't btw
    if eigenvalues.dtype == 'complex128':
        print("My dude, you got complex eigenvalues. Now I am not gonna break down, but you should totally give me higher sigmas (σ). (;")
        eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real

    # sort eigenvalues and vectors
    idx = np.argsort(eigenvalues)
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

    # SOLUTION A: Fiedler-vector solution
    # - consider only the SECOND smallest eigenvector
    # - threshold it at zero
    # - return as labels
    labels = eigenvectors[:, 1]
    if fiedler_solution:
        labels[labels < 0] = 0
        return labels

    # SOLUTION B: K-Means solution
    # - consider eigenvectors up to the n_cl-th
    # - use them as features instead of data for KMeans
    # - You want to use sklearn's implementation (;
    # - return KMeans' clusters

    new_features = eigenvectors[:, 1:n_cl + 1]
    labels = KMeans(n_cl).fit_predict(new_features)

    return labels


def main_spectral_clustering():
    """
    Main function for spectral clustering.
    """

    # generate the dataset
    #data, cl = two_moon_dataset(n_samples=300, noise=0.1)
    data, cl = gaussians_dataset(n_gaussian=3, n_points=[100, 100, 70], mus=[[1, 1], [-4, 6], [8, 8]], stds=[[1, 1], [3, 3], [1, 1]])

    # visualize the dataset
    _, ax = plt.subplots(1, 2)
    ax[0].scatter(data[:, 0], data[:, 1], c=cl, s=40)

    # run spectral clustering - tune n_cl and sigma!!!
    labels = spectral_clustering(data, n_cl=15, sigma=2, fiedler_solution=False)

    # visualize results
    ax[1].scatter(data[:, 0], data[:, 1], c=labels, s=40)
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main_spectral_clustering()
