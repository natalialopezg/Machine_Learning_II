# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================
# System-specific parameters and functions
import sys

# Data structure
import pandas as pd

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns

# Datasets
from sklearn.datasets import load_digits

# # Data preparation
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

## Model evaluation
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# ==============================================================================
#                          FUNCTIONS
# ==============================================================================

def training_model(X,y):
    # Training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, 
                                                        test_size=0.3, 
                                                        random_state = 123,
                                                        shuffle = True)

    # Create linear regression object
    # model = LogisticRegression()
    model = GaussianNB()

    # Train the model using the training sets
    model.fit(X_train, y_train.squeeze())

    # Make predictions using the testing set
    y_pred = model.predict(X_test)

    # Model evaluateion
    print(f"Accuracy_score: {accuracy_score(y_test, y_pred)}\n")

    # Confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)

    print(metrics.classification_report(y_test, y_pred))

    return conf_mat

# ==============================================================================
#                          WORKSHOP SOLUTION - POINT 6
# ==============================================================================

# 6) Now, apply dimensionality reduction using all your algorithms to train the 
# model with only 2 features per image
# ==============================================================================
# Path of unsupervised algorithms from scratch
package_path = r'LAB1\Unsupervised\dimensional_reduction'
sys.path.append(package_path) #Add custom classes path to script

# Load dataset
mnist = load_digits()

# Define arrays X: data matrix, y: classification target
X_raw = pd.DataFrame(mnist.data)
y_raw = pd.DataFrame(mnist.target)

# Filter data by selecting classes 0s and 8s
y = y_raw[y_raw[0].isin([0,8])]
X = X_raw.iloc[y.index]

print(f"Features of dataset: {mnist['feature_names']}\n")
print(f"Data matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Applying PCA on data
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
fig1 = plt.figure(1,figsize=(12,3)) 
fig1.add_subplot(1, 3, 1) 
plot = plt.scatter(X_transformed_pca[:,0], X_transformed_pca[:,1], c=y)
plt.title("Applying PCA")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(mnist['target_names']),
           loc='lower left')

# Applying SVD on raw data
# ==============================================================================
# Import implemented packages 
from svd import SVD

print("\nDimensionality Reduction with SVD")

# create a SVD object with 2 components
svd = SVD(n_components=2)

# fit the data
svd.fit(X)

# transform the data using the SVD object
X_transformed_svd = svd.transform(X)

print(f"Transformed data matrix shape: {X_transformed_svd.shape}")

# Plotting SVD results
fig1.add_subplot(1, 3, 2) 
plot = plt.scatter(X_transformed_svd[:,0], X_transformed_svd[:,1], c=y)
plt.title("Applying SVD")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(mnist['target_names']),
           loc='lower left')


# Applying T-SNE on raw data
# ==============================================================================
# Import implemented packages 
from tsne import TSNE

print("\nDimensionality Reduction with T-SNE")

# create a T-SNE object with 2 components
tsne = TSNE(n_dimensions=2)

# fit the data
tsne.fit(X.to_numpy())

# transform the data using the T-SNE object
X_transformed_tsne = tsne.transform(X,1000)

print(f"Transformed data matrix shape: {X_transformed_tsne.shape}")

# Plotting T-SNE results
fig1.add_subplot(1, 3, 3) 
plot = plt.scatter(X_transformed_tsne[:,0], X_transformed_tsne[:,1], c=y)
plt.title("Applying T-SNE")
plt.legend(handles=plot.legend_elements()[0], 
           labels=list(mnist['target_names']),
           loc='lower left')

fig1.canvas.manager.set_window_title('New features generated by algorithms')

# Training models
#-----------------------------------------------------------------------
algorithms = ['pca','svd','tsne']

for i, algorithm in enumerate(algorithms):
    print(f"\nMetrics for {algorithm.upper()}\n{'--'*15}")

    exec(f"X_string = X_transformed_{algorithm}")

    conf_mat = training_model(X_string,y)

    # Plotting PCA results
    fig2 = plt.figure(2,figsize=(12,3)) 
    fig2.add_subplot(1, 3, i+1) 
    plot = sns.heatmap(conf_mat
                    , annot=True
                    , fmt='d'
                    , cmap = 'mako'
                )
    plt.title(f"Confusion matrix {algorithm}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

fig2.canvas.manager.set_window_title('Confusion matrices for each implemented algorithm')

plt.show()

print("Dimensionality reduction influences the performance of models by \
increasing the values of their metrics to 98% or above.")

