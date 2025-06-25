import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

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
    if fiedler_solution and n_cl != 2:
        raise Exception("Cannot apply Fiedler to more than 2 clusters!")

    # compute distances
    #dist_matrix = cdist(data, data, metric='sqeuclidean')
    dist_matrix = ((np.expand_dims(data, 0) - np.expand_dims(data, 1)) ** 2).sum(-1)
    # compute affinity matrix
    affinity_matrix = np.exp(-dist_matrix / (sigma ** 2))

    # compute degree matrix
    degree_matrix = np.diag(affinity_matrix.sum(1))

    # compute laplacian
    laplacian_matrix = degree_matrix - affinity_matrix

    # compute eigenvalues and vectors
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
    # ensure we are not using complex numbers - you shouldn't btw
    if eigenvalues.dtype == 'complex128':
        print("My dude, you got complex eigenvalues. Now I am not gonna break down, but you should totally give me higher sigmas (σ). (;")
        eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real

    # sort eigenvalues and vectors
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues, eigenvectors = eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]

    # Fiedler-vector solution
    labels = eigenvectors[:, 1] > 0
    if fiedler_solution:
        return labels

    # KMeans solution
    new_features = eigenvectors[:, 1:n_cl + 1]
    labels = KMeans(n_cl, n_init='auto').fit_predict(new_features)

    return labels

