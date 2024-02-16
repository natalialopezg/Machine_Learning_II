# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================
# System-specific parameters and functions
import sys

# Graphics
import matplotlib.pyplot as plt

# Datasets
from sklearn.datasets import load_wine

# Standardization
from sklearn.preprocessing import StandardScaler

# ==============================================================================
#                          WORKSHOP SOLUTION - POINT 3
# ==============================================================================

# 3) Let’s create the unsupervised Python package
# ==============================================================================
# Path of unsupervised algorithms from scratch
package_path = r'LAB1\Unsupervised\dimensional_reduction'
sys.path.append(package_path) #Add custom classes path to script

# Load DataSet for test of implemented packages
wine_data = load_wine()

# Define arrays X: data matrix, y: classification target
X, y = wine_data['data'], wine_data['target']

print(f"Features of dataset: {wine_data['feature_names']}\n")
print(f"Data matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Plotting two random features of raw data
fig1 = plt.figure(1,figsize=(12,3)) 
fig1.add_subplot(1, 3, 1) 
plot = plt.scatter(X[:,0], X[:,2], c=y) # 0: 'alcohol', 2: 'ash', y -> 3 classes
plt.title("Raw data")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(wine_data['target_names']),
           loc='lower left')

# Applying PCA on raw data
# ==============================================================================
# Import implemented packages 
from pca import PCA

print("\nDimensionality Reduction with PCA")

# create a PCA object with 2 components
pca = PCA(n_components=2)

# fit the data
pca.fit(X)

# transform the data using the PCA object
X_transformed_pca = pca.transform(X)

print(f"Transformed data matrix shape: {X_transformed_pca.shape}")

# Plotting PCA results
fig1.add_subplot(1, 3, 2) 
plot = plt.scatter(X_transformed_pca[:,0], X_transformed_pca[:,1], c=y)
plt.title("Applying PCA on raw data")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(wine_data['target_names']),
           loc='lower left')

# Applying PCA on standardized data (PCA is sensitive to the scale!)
# ==============================================================================

# Normalise the data
scaler = StandardScaler()
scaler.fit(X)
X_normalized = scaler.transform(X)

# Apply PCA now
pca.fit(X_normalized)

# transform the data using the PCA object
X_transformed_pca = pca.transform(X_normalized)

# Plotting PCA results
fig1.add_subplot(1, 3, 3) 
plot = plt.scatter(X_transformed_pca[:,0], X_transformed_pca[:,1], c=y)
plt.title("Applying PCA on standardized data")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(wine_data['target_names']),
           loc='lower left')

fig1.tight_layout()
fig1.canvas.manager.set_window_title('Implement PCA from scratch')

# Applying SVD on raw data
# ==============================================================================
fig2 = plt.figure(2,figsize=(12,3)) 
fig2.add_subplot(1, 3, 1) 
plot = plt.scatter(X[:,0], X[:,2], c=y) # 0: 'alcohol', 2: 'ash', y -> 3 classes
plt.title("Raw data")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(wine_data['target_names']),
           loc='lower left')

# Import implemented packages 
from svd import SVD

print("\nDimensionality Reduction with SVD")


# create a SVD object with 2 components
svd = SVD(n_components=3)

# fit the data
svd.fit(X)

# transform the data using the SVD object
X_transformed_svd = svd.transform(X)

print(f"Transformed data matrix shape: {X_transformed_svd.shape}")

# Plotting SVD results
fig2.add_subplot(1, 3, 2) 
plot = plt.scatter(X_transformed_svd[:,0], X_transformed_svd[:,1], c=y)
plt.title("Applying SVD on raw data")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(wine_data['target_names']),
           loc='lower left')

# Applying SVD on standardized data 
# ==============================================================================

# Normalise the data
scaler = StandardScaler()
scaler.fit(X)
X_normalized = scaler.transform(X)

# Apply SVD now
svd.fit(X_normalized)

# transform the data using the SVD object
X_transformed_svd = svd.transform(X_normalized)

# Plotting SVD results
fig2.add_subplot(1, 3, 3) 
plot = plt.scatter(X_transformed_svd[:,0], X_transformed_svd[:,1], c=y)
plt.title("Applying SVD on standardized data")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(wine_data['target_names']),
           loc='lower left')

fig2.tight_layout()
fig2.canvas.manager.set_window_title('Implement SVD from scratch')

# Applying T-SNE on raw data
# ==============================================================================
fig3 = plt.figure(3,figsize=(12,3)) 
fig3.add_subplot(1, 3, 1) 
plot = plt.scatter(X[:,0], X[:,2], c=y) # 0: 'alcohol', 2: 'ash', y -> 3 classes
plt.title("Raw data")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(wine_data['target_names']),
           loc='lower left')

# Import implemented packages 
from tsne import TSNE

print("\nDimensionality Reduction with T-SNE")


# create a T-SNE object with 2 components
tsne = TSNE(n_dimensions=2)

# fit the data
tsne.fit(X)

# transform the data using the T-SNE object
X_transformed_tsne = tsne.transform(X,1000)

print(f"Transformed data matrix shape: {X_transformed_tsne.shape}")

# Plotting T-SNE results
fig3.add_subplot(1, 3, 2) 
plot = plt.scatter(X_transformed_tsne[:,0], X_transformed_tsne[:,1], c=y)
plt.title("Applying T-SNE on raw data")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(wine_data['target_names']),
           loc='lower left')

# Applying T-SNE on standardized data 
# ==============================================================================

# Normalise the data
scaler = StandardScaler()
scaler.fit(X)
X_normalized = scaler.transform(X)

# Apply T-SNE now
tsne.fit(X_normalized)

# transform the data using the T-SNE object
X_transformed_tsne = tsne.transform(X_normalized,1000)

# Plotting T-SNE results
fig3.add_subplot(1, 3, 3) 
plot = plt.scatter(X_transformed_tsne[:,0], X_transformed_tsne[:,1], c=y)
plt.title("Applying T-SNE on standardized data")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(wine_data['target_names']),
           loc='lower left')

fig3.tight_layout()
fig3.canvas.manager.set_window_title('Implement T-SNE from scratch')

plt.show()