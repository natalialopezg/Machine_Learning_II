# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================
# Multidimensional array object
import numpy as np

# Graphics
import matplotlib.pyplot as plt
from PIL import Image

# System-specific parameters and functions
import sys

# ==============================================================================
#                          WORKSHOP SOLUTION - POINT 4
# ==============================================================================

# 4) Apply SVD over the picture of your face, progressively increasing the 
# number of singular values used. Is there any point where you can say the 
# image is appropriately reproduced? How would you quantify how different 
# your photo and the approximation are?
# ==============================================================================
# Path of unsupervised algorithms from scratch
package_path = r'LAB1\Unsupervised\dimensional_reduction'
sys.path.append(package_path) #Add custom classes path to script

# Read picture
my_face = Image.open(r"LAB1\pictures\Natalia_Lopez.jpg").convert('L')
my_face_array = np.array(my_face,dtype= float)

# Applying SVD on picture of my face
# ==============================================================================
# Import implemented packages 
from svd import SVD

print("\nDimensionality Reduction with SVD\n")

distance = 50000 # 16.476 using n = 1
n = 0
stop_distance = 1000

while distance > stop_distance:
    # create a SVD object with n components
    svd = SVD(n_components=n)

    # fit the data
    svd.fit(my_face_array)

    # transform the data using the SVD object
    X_transformed_svd = svd.transform(my_face_array)

    # Create image from array
    approximation = Image.fromarray(X_transformed_svd.astype('uint8'),mode="L")

    # Distance from average
    appr_array = np.array(X_transformed_svd,dtype= float)

    diff =  appr_array - my_face_array

    distance = np.sqrt(np.sum(np.square(diff)))
    
    print(f"n_components: {n} | distance: {distance}") 

    n= n + 1

print(f"\nThe picture of my face is {distance:,.0f} pixels distant from picture \
approximation using n_components = {n-1} and is appropriately reproduced.\n")
print(f"\nQuantifying the difference between my photo and the approximation\
 is possible using Euclidean Distance mentioned in Point 2. ")

# Plot approximation face
fig = plt.figure(1,figsize=(10, 5)) 
fig.add_subplot(1, 2, 1) 
plt.imshow(my_face, cmap='gray') 
plt.title("Original picture")
fig.add_subplot(1, 2, 2) 
plt.imshow(approximation, cmap='gray') 
plt.title(f"Approximation picture n = {n-1}")
plt.show()
