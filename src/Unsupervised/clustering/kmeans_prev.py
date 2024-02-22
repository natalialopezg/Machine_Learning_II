# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================
# Multidimensional array object
import numpy as np

# ==============================================================================
#                                 FUNCTIONS
# ==============================================================================
# Create class
class KMeans:

    def __init__(self, n_clusters=8,max_iter = 300,tol=1e-5,random_state=None):
        self.K = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels_ = None
        self.random_state = random_state

    # randomly initializing K centroid by picking K samples from X
    def _initialize_random_centroids(self, X):
        """Initializes and returns k random centroids"""
        m, n = np.shape(X)
        # a centroid should be of shape (1, n), so the centroids array will be of shape (K, n)
        centroids = np.empty((self.K, n))
        for i in range(self.K):
            # pick a random data point from X as the centroid
            np.random.seed(self.random_state)
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
            cluster_idx[i] = self._closest_centroid(X[i], centroids)
        return cluster_idx

    # Compute the means of cluster to find new centroids.
    def _compute_means(self, cluster_idx, X):
        """Computes and returns the new centroids of the clusters"""
        _, n = np.shape(X)
        centroids = np.empty((self.K, n))
        for i in range(self.K):
            points = X[cluster_idx == i] # gather points for the cluster i
            centroids[i] = np.mean(points, axis=0) # use axis=0 to compute means across points
        return centroids

    #the K-means algorithm for the required number of iterations
    def fit(self,X):
        """Runs the K-means algorithm and computes the final clusters"""
        # initialize random centroids
        centroids = self._initialize_random_centroids(X)
        # loop till max_iterations or convergance
        print(f"initial centroids: {centroids}")
        for _ in range(self.max_iter):
            # create clusters by assigning the samples to the closet centroids
            clusters = self._create_clusters(centroids, X)
            previous_centroids = centroids                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
            # compute means of the clusters and assign to centroids
            centroids = self._compute_means(clusters, X)
            # if the new_centroids are the same as the old centroids, return clusters
            diff = previous_centroids - centroids
            # if not diff.any():
            #     return clusters
            # Check for convergence
            if diff.all() < self.tol:
                break
        self.cluster_centers_ = centroids
        self.labels_ = clusters
        # return clusters

    def predict(self, X):
        m, _ = np.shape(X)
        cluster_idx = np.empty(m)
        for i in range(m):
            cluster_idx[i] = self._closest_centroid(X[i], self.cluster_centers_)
        return cluster_idx



# ==============================================================================
#                              K-MEANS FROM SCRATCH
# ==============================================================================
# Graphics
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs 

X, true_labels = make_blobs(
    n_samples=750, centers=4, cluster_std=0.4, random_state=0
)

kmeans = KMeans(n_clusters=4, max_iter=1000)

# fit the data
kmeans.fit(X)

# kmeans.labels_
# kmeans.predict([[0, 0], [12, 3]])
# kmeans.cluster_centers_

print(kmeans.labels_)
plt.scatter(X[:, 0], X[:, 1], c= kmeans.labels_)
plt.show()