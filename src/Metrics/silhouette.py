# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================
# Graphics
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Multidimensional array object
import numpy as np

# System-specific parameters and functions
import sys

# Path of implemented algorithms from scratch
package_path = r'src'
sys.path.append(package_path) #Add custom classes path to script

# Clustering
from sklearn.cluster import KMeans

# Implemented K-means and K-medoids modules
from Unsupervised.clustering.kmeans import KMeans as KMeans_implemented
from Unsupervised.clustering.kmedoids import KMedoids as KMedoids_implemented

# Metrics
from sklearn.metrics import silhouette_samples, silhouette_score

# ==============================================================================
#                    SILHOUETTE ANALYSIS ON KMEANS CLUSTERING
# ==============================================================================
def plot(X,
        range_n_clusters,
        figure=True, 
        algorithm = 'sklearn-kmeans',
        algorithm_params={'n_clusters':8, 'max_iter':1000, 'random_state':None}):

    for n_clusters in range_n_clusters:

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 6))
        fig.canvas.manager.set_window_title(f"Silhouette n_clusters={n_clusters}")

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Execute de specified algorithm
        algorithm_params['n_clusters'] = n_clusters

        if algorithm == 'sklearn-kmeans':
            clusterer = KMeans(**algorithm_params)
        elif algorithm == 'implemented-kmeans':
            clusterer = KMeans_implemented(**algorithm_params)
        elif algorithm == 'implemented-kmedoids':
            clusterer = KMedoids_implemented(**algorithm_params)

        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette analysis for {algorithm} clustering with n_clusters = {n_clusters}",
            fontsize=14,
            fontweight="bold",
        )

    if figure == False:
        plt.close('all')
    else:
        plt.show(block=True)
