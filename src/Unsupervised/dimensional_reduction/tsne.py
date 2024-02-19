# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================
# Multidimensional array object
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

# ==============================================================================
#                             T-SNE FROM SCRATCH
# ==============================================================================
# Create class
class TSNE:
    def __init__(self,
        perplexity = 10,
        iterations = 1000, 
        Eta = 200,
        early_exaggeration = 4,
        n_dimensions = 2):

        self.n_dimensions = n_dimensions
        self.perplexity = perplexity
        self.iterations = iterations
        self.Eta = Eta
        self.early_exaggeration = early_exaggeration
        
    def _get_original_pairwise_affinities(self,X:np.array([])):

        '''
        Function to obtain affinities matrix.
        '''

        n = len(X)

        print("Computing Pairwise Affinities....")

        p_ij = np.zeros(shape=(n,n))
        for i in range(0,n):
            
            # Equation 1 numerator
            diff = X[i]-X
            
            sigma_i = self._grid_search(diff, i) # Grid Search for σ_i
            norm = np.linalg.norm(diff, axis=1)
            p_ij[i,:] = np.exp(-norm**2/(2*sigma_i**2))

            # Set p = 0 when j = i
            np.fill_diagonal(p_ij, 0)
            
            # Equation 1 
            p_ij[i,:] = p_ij[i,:]/np.sum(p_ij[i,:])

        # Set 0 values to minimum numpy value (ε approx. = 0) 
        ε = np.nextafter(0,1)
        p_ij = np.maximum(p_ij,ε)

        print("Completed Pairwise Affinities Matrix. \n")

        return p_ij

    def _grid_search(self,diff_i, i):

        '''
        Helper function to obtain σ's based on user-specified perplexity.
        '''

        result = np.inf # Set first result to be infinity

        norm = np.linalg.norm(diff_i, axis=1)
        std_norm = np.std(norm) # Use standard deviation of norms to define search space

        for sigma_search in np.linspace(0.01*std_norm,5*std_norm,200):

            # Equation 1 Numerator
            p = np.exp(-norm**2/(2*sigma_search**2)) 

            # Set p = 0 when i = j
            p[i] = 0 

            # Equation 1 (ε -> 0) 
            ε = np.nextafter(0,1)
            p_new = np.maximum(np.divide(p,np.sum(p)),ε) 
            
            # Shannon Entropy
            H = -np.sum(p_new*np.log2(p_new))
            
            # Get log(perplexity equation) as close to equality
            if np.abs(np.log(self.perplexity) - H * np.log(2)) < np.abs(result):
                result = np.log(self.perplexity) - H * np.log(2)
                sigma = sigma_search
        
        return sigma

    def _get_symmetric_p_ij(self, p_ij:np.array([])):

        '''
        Function to obtain symmetric affinities matrix utilized in t-SNE.
        '''
            
        print("Computing Symmetric p_ij matrix....")

        n = len(p_ij)
        p_ij_symmetric = np.zeros(shape=(n,n))
        for i in range(0,n):
            for j in range(0,n):
                p_ij_symmetric[i,j] = (p_ij[i,j] + p_ij[j,i]) / (2*n)
        
        # Set 0 values to minimum numpy value (ε approx. = 0)
        ε = np.nextafter(0,1)
        p_ij_symmetric = np.maximum(p_ij_symmetric,ε)

        print("Completed Symmetric p_ij Matrix. \n")

        return p_ij_symmetric

    def _initialization(self, X: np.array([])):

        return np.random.normal(loc=0,scale=1e-4,size=(len(X),self.n_dimensions))

    def _get_low_dimensional_affinities(self,Y:np.array([])):
        '''
        Obtain low-dimensional affinities.
        '''

        n = len(Y)
        q_ij = np.zeros(shape=(n,n))

        for i in range(0,n):

            # Equation 4 Numerator
            diff = Y[i]-Y
            norm = np.linalg.norm(diff, axis=1)
            q_ij[i,:] = (1+norm**2)**(-1)

        # Set p = 0 when j = i
        np.fill_diagonal(q_ij, 0)

        # Equation 4 
        q_ij = q_ij/q_ij.sum()

        # Set 0 values to minimum numpy value (ε approx. = 0)
        ε = np.nextafter(0,1)
        q_ij = np.maximum(q_ij,ε)

        return q_ij

    def _get_gradient(self,p_ij: np.array([]),
                    q_ij: np.array([]),
                    Y: np.array([])):
        '''
        Obtain gradient of cost function at current point Y.
        '''

        n = len(p_ij)

        # Compute gradient
        gradient = np.zeros(shape=(n, Y.shape[1]))
        for i in range(0,n):

            # Equation 5
            diff = Y[i]-Y
            A = np.array([(p_ij[i,:] - q_ij[i,:])])
            B = np.array([(1+np.linalg.norm(diff,axis=1))**(-1)])
            C = diff
            gradient[i] = 4 * np.sum((A * B).T * C, axis=0)

        return gradient 

    def fit_transform(self, X):
        # Get original affinities matrix 
        p_ij = self._get_original_pairwise_affinities(X)
        self.p_ij_symmetric = self._get_symmetric_p_ij(p_ij)

        n = len(X)
        # Initialization
        Y = np.zeros(shape=(self.iterations, n, self.n_dimensions))
        Y_minus1 = np.zeros(shape=(n, self.n_dimensions))
        Y[0] = Y_minus1
        Y1 = self._initialization(X)
        Y[1] = np.array(Y1)

        print("Optimizing Low Dimensional Embedding....")
        # Optimization
        for t in range(1, self.iterations-1):
            
            # Momentum & Early Exaggeration
            if t < 250:
                alfa = 0.5
                self.early_exaggeration = self.early_exaggeration
            else:
                alfa = 0.8
                self.early_exaggeration = 1

            # Get Low Dimensional Affinities
            q_ij = self._get_low_dimensional_affinities(Y[t])

            # Get Gradient of Cost Function
            gradient = self._get_gradient(self.early_exaggeration*self.p_ij_symmetric, q_ij, Y[t])

            # Update Rule
            Y[t+1] = Y[t] - self.Eta * gradient + alfa * (Y[t] - Y[t-1]) # Use negative gradient 

            # Compute current value of cost function
            if t % 50 == 0 or t == 1:
                cost = np.sum(self.p_ij_symmetric * np.log(self.p_ij_symmetric / q_ij))
                print(f"Iteration {t}: Value of Cost Function is {cost}")

        print(f"Completed Embedding: Final Value of Cost Function is {np.sum(self.p_ij_symmetric * np.log(self.p_ij_symmetric / q_ij))}")
        solution= Y[-1]  

        return solution
