# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================

# Multidimensional array object
import numpy as np

# ==============================================================================
#                              PCA FROM SCRATCH
# ==============================================================================
# Create class
class KMedoids:
    def __init__(self, n_clusters = 8, max_iter = 300,random_state = None):
        ''' 
        Class constructor
        Parameters
        ----------
        - k: number of clusters. 
        - max_iter: number of times centroids will move
        - has_converged: to check if the algorithm stop or not
        '''
        self.k = n_clusters
        self.max_iter = max_iter
        self.has_converged = False
        self.medoids_cost = None
        self.random_state = random_state
        self.cluster_centers_= None
        self.labels = None

    def _euclideanDistance(self,x, y):
        '''
        Euclidean distance between x, y
        --------
        Return
        d: float
        '''
        # d = np.sqrt(np.sum(np.power(x - y, 2))) 
        d = np.linalg.norm(x - y) 

        return d
        
    def _initialize_medoids(self, X):
        ''' 
        Parameters
        ----------
        X: input data. 
        '''
        self.cluster_centers_= []

        # Reseed the singleton RandomState instance
        np.random.seed(self.random_state)
        
        # Initialize centroids randomly
        idx = np.random.choice(X.shape[0], self.k, replace=False)
        self.cluster_centers_= X[idx]
        
        self.medoids_cost = [0] * self.k

    def _update_medoids(self, X, labels):
        '''
        Parameters
        ----------
        labels: a list contains labels of data points
        '''
        self.has_converged = True
        
        #Store data points to the current cluster they belong to
        clusters = []

        # for each cluster
        for j in range(0,self.k):
            cluster = []

            # for each point in X
            for i in range(X.shape[0]):
                if (labels[i] == j):
                    cluster.append(X[i])
            clusters.append(cluster)
        
        #Calculate the new medoids
        new_medoids = []
        for i in range(0, self.k):
            new_medoid = self.cluster_centers_[i]
            old_medoids_cost = self.medoids_cost[i]

            for j in range(len(clusters[i])):
                #Cost of the current data points to be compared with the current optimal cost
                cur_medoids_cost = 0
                for dpoint_index in range(len(clusters[i])):
                    cur_medoids_cost += self._euclideanDistance(clusters[i][j], clusters[i][dpoint_index])
                
                #If current cost is less than current optimal cost,
                #make the current data point new medoid of the cluster
                if cur_medoids_cost < old_medoids_cost:
                    new_medoid = clusters[i][j]
                    old_medoids_cost = cur_medoids_cost
            
            #Now we have the optimal medoid of the current cluster
            new_medoids.append(new_medoid)
        
        #If not converged yet, accept the new medoids
        if not set([tuple(x) for x in self.cluster_centers_]) == set([tuple(x) for x in new_medoids]):
            self.cluster_centers_= new_medoids
            self.has_converged = False
    
    def fit(self, X):
        '''
        FIT function, used to find clusters
        Parameters
        ----------
        X: input data. 
        '''

        self._initialize_medoids(X)
        
        for iter in range(self.max_iter):
            #Labels for this iteration
            labels = []

            # for each cluster
            for medoid in range(0,self.k):
                # Initialize dissimilarity cost of the current cluster
                self.medoids_cost[medoid] = 0

                # for each point in X
                for i in range(X.shape[0]):
                    #Distances from a data point to each of the medoids
                    point_distances = []

                    # Calculate the distance from each cluster to point i                  
                    for j in range(0,self.k):
                        point_distances.append(self._euclideanDistance(self.cluster_centers_[j], X[i]))

                    #Data points' label is the medoid which has minimal distance to it
                    labels.append(point_distances.index(min(point_distances)))
                    
                    self.medoids_cost[medoid] += min(point_distances)

            self._update_medoids(X, labels)
            
            if self.has_converged:
                break

        self.cluster_centers_ = np.array([array.tolist() for array in self.cluster_centers_])
        
        
    def fit_predict(self,data):
        ''' 
        Parameters
        ----------
        data: input data.
        
        Returns:
        ----------
        pred: list cluster indexes of input data 
        '''
        self.fit(data)
    
        pred = []
        for i in range(len(data)):
            #Distances from a data point to each of the medoids
            d_list = []
            for j in range(len(self.cluster_centers_)):
                d_list.append(self._euclideanDistance(self.cluster_centers_[j],data[i]))
                
            pred.append(d_list.index(min(d_list)))
            
        return np.array(pred)
