# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================
# System-specific parameters and functions
import sys

# Path of implemented algorithms from scratch
package_path = r'src'
sys.path.append(package_path) #Add custom classes path to script

# Graphics
import matplotlib.pyplot as plt

# Clustering
from sklearn.cluster import KMeans, DBSCAN, spectral_clustering, SpectralClustering
from sklearn_extra.cluster import KMedoids

# Metrics
from sklearn import metrics
from Metrics import silhouette # Import implemented package

# ==============================================================================
#                                 FUNCTIONS
# ==============================================================================
count_fig = 0

def scatterplots(data,list_datasets,title='title'):
    '''
    Returns the scatterplots figure build from input parameters.

      Parameters:
                data              [list[tuple]]       List of tuples of datasets
                list_datasets     [list(strings)]     List of names of datasets
                title             [string]            Title for figure

      Returns:
              Plot of figure
    '''
    global count_fig

    # visualization_dimensions
    nrows = int(np.floor(len(list_datasets)**0.5))
    ncols = int(np.ceil(len(list_datasets) / nrows))

    # Create figure
    count_fig = count_fig + 1
    fig = plt.figure(count_fig,figsize=((ncols*3)+2, (nrows*3)+1))
    title_dist =1

    for i, dataset_name in enumerate(list_datasets):

        fig.add_subplot(nrows,ncols, i+1)
        plot = plt.scatter(data[i][0][:,0], data[i][0][:,1], c=data[i][1])
        plt.title(dataset_name)
        plt.legend(handles=plot.legend_elements()[0],
                    labels=list(np.unique(data[i][1])),
                    loc='lower right')

    fig.tight_layout()
    plt.subplots_adjust(top = 0.9)
    fig.suptitle(title, fontsize = 15, fontweight = "bold", y= title_dist)

    plt.show()

def clustering(X,y, kmeans_param,kmedoids_param,DBSCAN_param,SpectralClustering_param,title):
    '''
    Returns the scatterplots figure build of clustering algorithms

      Parameters:
                X                           [sparse matrix]    Training instances to cluster
                kmeans_param                [dict]             Dict of parameters for k-means
                kmedoids_param              [dict]             Dict of parameters for k-medoids
                DBSCAN_param                [dict]             Dict of parameters for DBSCAN
                SpectralClustering_param    [dict]             Dict of parameters for SpectralClustering
                title                       [string]           Title for figure

      Returns:
              Plot of figure
    '''
    # K-means
    kmeans = KMeans(**kmeans_param).fit(X)
    kmeans_output = (X,kmeans.labels_)

    # K-medoids
    kmedoids = KMedoids(**kmedoids_param).fit(X)
    kmedoids_output = (X,kmedoids.labels_)

    # DBSCAN
    DBSCAN_estimator = DBSCAN(**DBSCAN_param).fit(X)
    DBSCAN_output = (X,DBSCAN_estimator.labels_)

    # SpectralClustering
    SpectralClustering_estimator = SpectralClustering(**SpectralClustering_param).fit(X)
    SpectralClustering_output = (X,SpectralClustering_estimator.labels_)

    # Metrics
    if type(y) != type(None):
        kmeans_rand_score = metrics.rand_score(y, kmeans.labels_)
        kmedoids_rand_score = metrics.rand_score(y, kmedoids.labels_)
        DBSCAN_rand_score = metrics.rand_score(y, DBSCAN_estimator.labels_)
        SpectralClustering_rand_score = metrics.rand_score(y, SpectralClustering_estimator.labels_)
    else:
        kmeans_rand_score = 0
        kmedoids_rand_score = 0
        DBSCAN_rand_score = 0
        SpectralClustering_rand_score = 0

    # Plot the different clustering algorithms
    data = [kmeans_output,kmedoids_output,DBSCAN_output,SpectralClustering_output]
    list_datasets = [f"K-means | rand_score = {kmeans_rand_score:.3f}",
                     f"K-medoids | rand_score = {kmedoids_rand_score:.3f}",
                     f"DBSCAN | rand_score = {DBSCAN_rand_score:.3f}",
                     f"SpectralClustering | rand_score = {SpectralClustering_rand_score:.3f}"]
    scatterplots(data,list_datasets,f"{title} clustering")

# ==============================================================================
#                       WORKSHOP II SOLUTION - POINT 6
# ==============================================================================

# 6) Use the following code snippet to create different types of scattered data:
# ==============================================================================
# Code
import numpy as np
from sklearn import cluster, datasets, mixture
# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = (np.random.rand(n_samples, 2), None)
# Anisotropically distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)
# blobs with varied variances
varied = datasets.make_blobs(
n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

# Plot the different datasets in separate figures
# ==============================================================================
data = [noisy_circles,noisy_moons,blobs,no_structure,aniso,varied]
list_datasets = ['noisy_circles','noisy_moons','blobs','no_structure','aniso','varied']
scatterplots(data,list_datasets,'Plotting of different scattered data')

# Apply k-means, k-medoids, DBSCAN and Spectral Clustering from Scikit-Learn
# over each dataset
# ==============================================================================

# noisy_circles dataset
# ============================================
# data
X = noisy_circles[0]
y = noisy_circles[1]

# Definition of clustering algorithm parameters
kmeans_param = {'n_clusters':2, 'random_state':0}
kmedoids_param = {'n_clusters':2, 'random_state':0}
DBSCAN_param  = {'eps': 0.2}
SpectralClustering_param = {'n_clusters':2,'affinity':'nearest_neighbors','random_state':0}

# Clustering
clustering(X,y, kmeans_param,kmedoids_param,DBSCAN_param,SpectralClustering_param,'noisy_circles')

# noisy_moons dataset
# ============================================
# data
X = noisy_moons[0]
y = noisy_moons[1]

# Definition of clustering algorithm parameters
kmeans_param = {'n_clusters':2, 'random_state':0}
kmedoids_param = {'n_clusters':2, 'random_state':0}
DBSCAN_param  = {'eps': 0.2}
SpectralClustering_param = {'n_clusters':2,'affinity':'nearest_neighbors','random_state':0}

# Clustering
clustering(X, y, kmeans_param,kmedoids_param,DBSCAN_param,SpectralClustering_param,'noisy_moons')

# blobs dataset
# ============================================
# data
X = blobs[0]
y = blobs[1]

# Definition of clustering algorithm parameters
kmeans_param = {'n_clusters':3, 'random_state':0}
kmedoids_param = {'n_clusters':3, 'random_state':0}
DBSCAN_param  = {'eps': 2}
SpectralClustering_param = {'n_clusters':3,'affinity':'nearest_neighbors','random_state':0}

# Clustering
clustering(X, y, kmeans_param,kmedoids_param,DBSCAN_param,SpectralClustering_param,'blobs')

# no_structure dataset
# ============================================
# data
X = no_structure[0]
y = no_structure[1]

# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8 ]
silhouette.plot(X,range_n_clusters,False)

# Definition of clustering algorithm parameters
kmeans_param = {'n_clusters':4, 'random_state':0}
kmedoids_param = {'n_clusters':4, 'random_state':0}
DBSCAN_param  = {'eps': 0.07}
SpectralClustering_param = {'n_clusters':4,'affinity':'nearest_neighbors','random_state':0}

# Clustering
clustering(X, y, kmeans_param,kmedoids_param,DBSCAN_param,SpectralClustering_param,'no_structure')

# aniso dataset
# ============================================
# data
X = aniso[0]
y = aniso[1]

# Definition of clustering algorithm parameters
kmeans_param = {'n_clusters':3, 'random_state':0}
kmedoids_param = {'n_clusters':3, 'random_state':0}
DBSCAN_param  = {'eps': 0.5}
SpectralClustering_param = {'n_clusters':3,'affinity':'nearest_neighbors','random_state':0}

# Clustering
clustering(X, y, kmeans_param,kmedoids_param,DBSCAN_param,SpectralClustering_param,'aniso')

# varied dataset
# ============================================
# data
X = varied[0]
y = varied[1]

# Definition of clustering algorithm parameters
kmeans_param = {'n_clusters':3, 'random_state':0}
kmedoids_param = {'n_clusters':3, 'random_state':0}
DBSCAN_param  = {'eps': 1, 'min_samples': 50}
SpectralClustering_param = {'n_clusters':3,'affinity':'nearest_neighbors','random_state':0}

# Clustering
clustering(X, y, kmeans_param, kmedoids_param, DBSCAN_param, SpectralClustering_param, 'varied')
