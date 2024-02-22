# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================
# System-specific parameters and functions
import sys

# Multidimensional array object
import numpy as np

# Graphics
import matplotlib.pyplot as plt

# Path of unsupervised algorithms from scratch
package_path = r'src'
sys.path.append(package_path) #Add custom classes path to script

# Implemented modules
from Metrics import silhouette 

# ==============================================================================
#                          WORKSHOP SOLUTION - POINT 5
# ==============================================================================

# a) Use the following code snippet to create scattered data X
# ==============================================================================
from sklearn.datasets import make_blobs
X, y = make_blobs(
n_samples=500,
n_features=2,
centers=4,
cluster_std=1,
center_box=(-10.0, 10.0),
shuffle=True,
random_state=1,
)

# b) Plot the resulting dataset. How many clusters are there? 
# How far are they from one another?
# ==============================================================================
plot = plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Dataset")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(np.unique(y)),
           loc='lower right')

plt.show()

print(f"\nHow many clusters are there? How far are \
they from one another?\n{'--'*30}\n\
There are 4 clusters in the dataset. \nOne of them is \
completely far away from the others, while the rest are closer, \
especially two of them that overlap slightly.\n")

# c) For both k-means and k-medoids (your implementations), 
# calculate the silhouette plots and coefficients for each run, 
# iterating K from 1 to 5 clusters
# ==============================================================================

# KMeans implemented
# ============================================
print(f"Silhouette analysis using implemeted algorithm KMeans\n{'--'*30}")
# Silhouette analysis
range_n_clusters = [2, 3, 4, 5]
silhouette.plot(X,
                range_n_clusters, 
                algorithm='implemented-kmeans',
                algorithm_params={'random_state':123})

# KMedoids implemented
# ============================================
print(f"\nSilhouette analysis using implemeted algorithm KMedoids\n{'--'*30}")
# Silhouette analysis
range_n_clusters = [2, 3, 4, 5]
silhouette.plot(X,
                range_n_clusters, 
                algorithm='implemented-kmedoids',
                algorithm_params={'random_state':123})