# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================

# Multidimensional array object
import numpy as np

# ==============================================================================
#                           SVD FROM SCRATCH
# ==============================================================================
# Create class
class SVD:
    
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # Calculate Singular-Value Decomposition
        self.U, self.S, self.Vt = np.linalg.svd(X)

        # Number of components to retain in decomposed matrix
        self.S = self.S[:self.n_components]
        self.Vt = self.Vt[:self.n_components, :]
        self.U = self.U[:, :self.n_components]


    def transform(self, X):
        self.X_transformed = np.dot(self.U,np.dot(np.diag(self.S),self.Vt))

        return self.X_transformed