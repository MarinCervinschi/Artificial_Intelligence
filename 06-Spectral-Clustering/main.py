import numpy as np
from datasets import two_moon_dataset, gaussians_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from spectral_clustering import spectral_clustering

def run_two_moon_clustering():
    """
    Run spectral clustering on the two moon dataset.
    """
    data, cl = two_moon_dataset(n_samples=300, noise=0.1)
    run_and_visualize_clustering(data, cl, n_cl=2, sigma=0.1)

def run_gaussians_clustering():
    """
    Run spectral clustering on the gaussians dataset.
    """
    data, cl = gaussians_dataset(
        n_gaussian=3,
        n_points=[100, 100, 70],
        mus=[[1, 1], [-4, 6], [8, 8]],
        stds=[[1, 1], [3, 3], [1, 1]]
    )
    run_and_visualize_clustering(data, cl, n_cl=3, sigma=1.0)

def run_and_visualize_clustering(data, cl, n_cl, sigma):
    """
    Helper function to run spectral clustering and visualize results.
    """
    _, ax = plt.subplots(1, 2)
    ax[0].scatter(data[:, 0], data[:, 1], c=cl, s=40)
    labels = spectral_clustering(data, n_cl=n_cl, sigma=sigma, fiedler_solution=n_cl==2)
    ax[1].scatter(data[:, 0], data[:, 1], c=labels, s=40)

    cm = confusion_matrix(cl, labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    labels_aligned = np.zeros_like(labels)
    for i, j in zip(col_ind, row_ind):
        labels_aligned[labels == i] = j

    acc = accuracy_score(cl, labels_aligned)
    print(f"Accuracy: {acc:.3f}")

    plt.waitforbuttonpress()

if __name__ == "__main__":
    # Uncomment the one you want to run:
    run_two_moon_clustering()
    #run_gaussians_clustering()
