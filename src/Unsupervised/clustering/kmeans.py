# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================
# Multidimensional array object
import numpy as np

# ==============================================================================
#                             K-MEANS FROM SCRATCH
# ==============================================================================
# Create class
class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.labels = None
        self.random_state = random_state

    def _calc_distances(self, X):
        # Calculates and returns the distances by taking the L2 norm 
        # of the difference between the two vectors
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.cluster_centers_):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances    

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Reseed the singleton RandomState instance
        np.random.seed(self.random_state)
        
        # Initialize centroids randomly
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx]
        
        for i in range(self.max_iter):
            # Assign each data point to the nearest centroid
            distances = self._calc_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for j in range(self.n_clusters):
                new_centroids[j] = np.mean(X[self.labels == j], axis=0)
                
            # Check for convergence
            if np.sum(np.abs(new_centroids - self.cluster_centers_)) < self.tol:
                break
                
            self.cluster_centers_ = new_centroids
            
    def fit_predict(self, X):
        self.fit(X)
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)

# Datasets
from sklearn.datasets import make_blobs 

# blobs dataset
# ============================================
blobs  = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=0)

# data
X = blobs[0]
y = blobs[1]

# KMedoids
# ============================================
kmeans = KMeans(n_clusters=4, max_iter=1000, random_state=123)

# fit the data
# kmedoids.fit(X)

# predict
y_kmeans= kmeans.fit_predict(X)
