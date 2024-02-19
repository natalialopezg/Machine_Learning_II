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
A = np.random.rand(4,5)
print(f"Random matrix 'A' of {A.shape[0]}x{A.shape[1]} shape: \n{A}\n")

# The rank of matrix A
rank_A = np.linalg.matrix_rank(A)
print(f"The rank of the matrix A is {rank_A}\n")

# print("**The following properties and transformations cannot be calculated \
# because the matrix A has a square form.**\n")

# The trace of matrix A
# trace_A = np.trace(A)
# print(f"The trace of the matrix A is {trace_A}\n")
print(f"The trace of matrix A cannot be calculated since it is not a square matrix.\n")

# The determinant of matrix A
# determinant_A = np.linalg.det(A)
# print(f"The determinant of the matrix A is {determinant_A}\n")
print(f"The determinant of matrix A cannot be calculated since it is not a square matrix.\n")

print(f"The inverse of the matrix A\n{'--'*30}")
# inverse_A = np.linalg.inv(A)
# print(f"The inverse of the matrix A is: \n{inverse_A}\n")
print("Non-square matrices, i.e. m-by-n matrices for which m ≠ n, \
do not have an inverse. However, in some cases such a matrix may  \
have a left inverse or right inverse. If A is m-by-n and the rank \
of A is equal to n, (n ≤ m), then A has a left inverse, an n-by-m \
matrix B such that BA = In. If A has rank m (m ≤ n), then it has  \
a right inverse, an n-by-m matrix B such that AB = Im.\nThe pinv \
function computes a pseudo inverse with computacional loss.\n")

print(f"Eigenvalues and eigenvectors of ATA and AAT\n{'--'*30}")
print("The definition of an eigenvalue is for square matrices.\
For non-square matrices, we can define singular values.\nThe \
singular values of a mxn matrix A are the positive square roots \
of the nonzero eigenvalues of the corresponding matrix ATA. The \
corresponding eigenvectors are called the singular vectors.\n")

A_transpose = np.transpose(A)  

# A transpose • Matrix A 
ATA = np.dot(A_transpose, A)
eigenvalues_ATA, eigenvector_ATA = np.linalg.eig(ATA)

# Matrix A • A transpose
AAT = np.dot(A, A_transpose)
eigenvalues_AAT, eigenvector_AAT = np.linalg.eig(AAT)

print(f"Eigenvalues\n{'--'*30}")
print(f"The eigenvalues of the matrix ATA: \n{eigenvalues_ATA}\n")

print(f"The eigenvalues of the matrix AAT: \n{eigenvalues_AAT}\n")

print("As for the eigenvalues, it is observed that the first component \
of both eigenvalue vectors is the same, the second component differs by \
an order of magnitude, while the other components differ by about an \
order of magnitude and may have different values.\n")

print(f"Eigenvectors\n{'--'*30}")
print(f"The eigenvectors of the matrix ATA: \n{eigenvector_ATA}\n")

print(f"The eigenvectors of the matrix AAT: \n{eigenvector_AAT}\n")

print("As for the eigenvectors, it can be seen that they are different \
matrices with different values.")