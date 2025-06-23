import numpy as np
from datasets import two_moon_dataset, gaussians_dataset
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from spectral_clustering import spectral_clustering

def main_spectral_clustering():
    """
    Main function for spectral clustering.
    """

    # generate the dataset
    data, cl = two_moon_dataset(n_samples=300, noise=0.1)
    #data, cl = gaussians_dataset(n_gaussian=3, n_points=[100, 100, 70], mus=[[1, 1], [-4, 6], [8, 8]], stds=[[1, 1], [3, 3], [1, 1]])

    # visualize the dataset
    _, ax = plt.subplots(1, 2)
    ax[0].scatter(data[:, 0], data[:, 1], c=cl, s=40)

    # run spectral clustering - tune n_cl and sigma!!!
    labels = spectral_clustering(data, n_cl=2, sigma=0.1, fiedler_solution=False)

    # visualize results
    ax[1].scatter(data[:, 0], data[:, 1], c=labels, s=40)
    
    # print accuracy
    # Calcola la migliore corrispondenza tra etichette
    cm = confusion_matrix(cl, labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    labels_aligned = np.zeros_like(labels)
    for i, j in zip(col_ind, row_ind):
        labels_aligned[labels == i] = j

    # Calcola e stampa l'accuracy
    acc = accuracy_score(cl, labels_aligned)
    print(f"Accuracy: {acc:.3f}")

    plt.waitforbuttonpress()


if __name__ == "__main__":
    main_spectral_clustering()
