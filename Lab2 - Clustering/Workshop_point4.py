# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================
# System-specific parameters and functions
import sys

# Multidimensional array object
import numpy as np

# Graphics
import matplotlib.pyplot as plt

# Datasets
from sklearn.datasets import make_blobs 

# Metrics
from sklearn import metrics

# Path of unsupervised algorithms from scratch
package_path = r'src'
sys.path.append(package_path) #Add custom classes path to script

# Implemented K-means and K-medoids modules
from Unsupervised.clustering.kmeans import KMeans
from Unsupervised.clustering.kmedoids import KMedoids

# ==============================================================================
#                          WORKSHOP SOLUTION - POINT 4
# ==============================================================================

# 4) Implementing k-means and k-medoids modules using Python and Numpy
# ==============================================================================

# blobs dataset
# ============================================
blobs  = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=0)

# data
X = blobs[0]
y = blobs[1]

# KMeans 
# ============================================
kmeans = KMeans(n_clusters=4, max_iter=1000, random_state=123)

# # fit the data
# kmeans.fit(X)

# predict
y_kmeans = kmeans.fit_predict(X)

# Metrics
kmeans_rand_score = metrics.rand_score(y, y_kmeans)

# KMedoids
# ============================================
kmedoids = KMedoids(n_clusters=4, max_iter=1000, random_state=123)

# fit the data
# kmedoids.fit(X)

# predict
y_kmedoids = kmedoids.fit_predict(X)

# Metrics
kmedoids_rand_score = metrics.rand_score(y, y_kmedoids)

# Plotting results
# ============================================
fig1 = plt.figure(1,figsize=(12,5)) 
fig1.suptitle(f"K-Means and K-Medoids from scratch",
             fontsize = 15, 
             fontweight = "bold", 
             y= 1)

fig1.add_subplot(1, 3, 1) 
plot = plt.scatter(X[:,0], X[:,1], c=y)
plt.title("True labels")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(np.unique(y)),
           loc='lower left')

fig1.add_subplot(1, 3, 2) 
plot = plt.scatter(X[:,0], X[:,1], c=y_kmeans)
plt.title(f"KMeans | rand_score= {kmeans_rand_score:.5f}")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(np.unique(y_kmeans)),
           loc='lower left')
    
fig1.add_subplot(1, 3, 3) 
plot = plt.scatter(X[:,0], X[:,1], c=y_kmedoids)
plt.title(f"KMedoids | rand_score= {kmedoids_rand_score:.5f}")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(np.unique(y_kmedoids)),
           loc='lower left')

fig1.tight_layout()
plt.show()