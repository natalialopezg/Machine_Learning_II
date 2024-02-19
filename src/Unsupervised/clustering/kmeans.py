# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================
# Multidimensional array object
import numpy as np

# ==============================================================================
#                                 FUNCTIONS
# ==============================================================================
# Create class
class PCA:

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        # self.components = None
        # self.mean = None

    # randomly initializing K centroid by picking K samples from X
    def initialize_random_centroids(self, X=None):
        """Initializes and returns k random centroids"""
        m, n = np.shape(X)
        # a centroid should be of shape (1, n), so the centroids array will be of shape (K, n)
        centroids = np.empty((self.K, n))
        for i in range(self.K):
            # pick a random data point from X as the centroid
            centroids[i] =  X[np.random.choice(range(m))] 
        return centroids

    # Calculate euclidean distance between two vectors
    def _euclidean_distance(self,x1, x2):
        """Calculates and returns the euclidean distance between two vectors x1 and x2"""
        return np.sqrt(np.sum(np.power(x1 - x2, 2))) # np.linalg.norm(x1 - x2)

    # Finding the closest centroid to a given data point
    def _closest_centroid(self,x, centroids):
        """Finds and returns the index of the closest centroid for a given vector x"""
        distances = np.empty(self.K)
        for i in range(self.K):
            distances[i] = self._euclidean_distance(centroids[i], x)
        return np.argmin(distances) # return the index of the lowest distance

    # Assign the samples to closest centroids to create the clusters
    def _create_clusters(self, centroids, X):
        """Returns an array of cluster indices for all the data samples"""
        m, _ = np.shape(X)
        cluster_idx = np.empty(m)
        for i in range(m):
            cluster_idx[i] = self._closest_centroid(X[i], centroids, self.K)
        return cluster_idx

    # Compute the means of cluster to find new centroids.
    def _compute_means(cluster_idx, X):
        """Computes and returns the new centroids of the clusters"""
        _, n = np.shape(X)
        centroids = np.empty((K, n))
        for i in range(K):
            points = X[cluster_idx == i] # gather points for the cluster i
            centroids[i] = np.mean(points, axis=0) # use axis=0 to compute means across points
        return centroids

#the K-means algorithm for the required number of iterations
def _run_Kmeans(K, X, max_iterations=500):
    """Runs the K-means algorithm and computes the final clusters"""
    # initialize random centroids
    centroids = initialize_random_centroids(K, X)
    # loop till max_iterations or convergance
    print(f"initial centroids: {centroids}")
    for _ in range(max_iterations):
        # create clusters by assigning the samples to the closet centroids
        clusters = create_clusters(centroids, K, X)
        previous_centroids = centroids                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        # compute means of the clusters and assign to centroids
        centroids = compute_means(clusters, K, X)
        # if the new_centroids are the same as the old centroids, return clusters
        diff = previous_centroids - centroids
        if not diff.any():
            return clusters
    return clusters


# ==============================================================================
#                              K-MEANS FROM SCRATCH
# ==============================================================================
# Create class
from sklearn import datasets
# creating a dataset for clustering
X, y = datasets.make_blobs()
y_preds = run_Kmeans(3, X)

from mlfromscratch.utils import Plot
p = Plot()
p.plot_in_2d(X, y_preds, title="K-Means Clustering")
p.plot_in_2d(X, y, title="Actual Clustering")