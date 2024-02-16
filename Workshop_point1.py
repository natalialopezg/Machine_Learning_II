# ==============================================================================
#                               LIBRARY IMPORT
# ==============================================================================

# Multidimensional array object
import numpy as np

# ==============================================================================
#                           WORKSHOP SOLUTION - POINT 1
# ==============================================================================

# 1) Simulate any random rectangular matrix A:
# ==============================================================================
# Matrix A
A = np.random.rand(4,4) # ****************
# A = np.random.randint(1,20,(4,4)) # ***********************
print(f"Random matrix 'A' of {A.shape[0]}x{A.shape[1]} shape: \n{A}\n")

# The rank of matrix A
rank_A = np.linalg.matrix_rank(A)
print(f"The rank of the matrix A is {rank_A}\n")

print("**The following properties and transformations are possible only \
because the matrix A has square form**\n")

# The trace of matrix A
trace_A = np.trace(A)
print(f"The trace of the matrix A is {trace_A}\n")

# The determinant of matrix A
determinant_A = np.linalg.det(A)
print(f"The determinant of the matrix A is {determinant_A}\n")

# A = np.array([[4,7,2],[2,6,3],[7,1,3]]) # ****************

# The inverse of the matrix A
print("The matrix A is invertible because it is a non-singular square \
matrix (its determinant is not zero).\nUsing numpy.linalg.inv: Given a\
square matrix A, return the matrix ainv satisfying \
dot(a, ainv) = dot(ainv, a) = eye(a.shape[0]).\n")
inverse_A = np.linalg.inv(A)
print(f"The inverse of the matrix A is: \n{inverse_A}\n")

# Eigenvalues and eigenvectors of A’A and AA’
# Inverse A • Matrix A 
ATA = np.dot(inverse_A, A)
eigenvalues_ATA, eigenvector_ATA = np.linalg.eig(ATA)

# Matrix A • Inverse A
AAT = np.dot(A, inverse_A)
eigenvalues_AAT, eigenvector_AAT = np.linalg.eig(AAT)

print(f"The eigenvalues of the matrix ATA: \n{eigenvalues_ATA}\n")

print(f"The eigenvalues of the matrix AAT: \n{eigenvalues_AAT}\n")

print(f"The eigenvectors of the matrix ATA: \n{eigenvector_ATA}\n")

print(f"The eigenvectors of the matrix AAT: \n{eigenvector_AAT}\n")

print("Depending on the values of the initial matrix, between \
eigenvalues of A'A and AA' there is no difference or eigenvalues \
of A'A may contain complex numbers. However, the real part of these\
coincides with eigenvalues of AA' (in most cases).\n\
In the case of eigenvectors the result is different. The eigenvectors \
of A'A contain complex numbers and eigenvectors of AA' contain\
real numbers regardless of the values of the initial matrix (in most cases).")
