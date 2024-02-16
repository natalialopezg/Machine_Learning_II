#Robust PCA factors a matrix into the sum of two matrices, M=L+S, where M is the original matrix, L is low-rank, and S is sparse. This is what we'll be using for the background removal problem! Low-rank means that the matrix has a lot of redundant information-- in this case, it's the background, which is the same in every scene (talk about redundant info!). Sparse means that the matrix has mostly zero entries-- in this case, see how the picture of the foreground (the people) is mostly empty. (In the case of corrupted data, S is capturing the corrupted entries).

# Robust PCA factors a matrix into the sum of two matrices, M=L+S, 
# where M is the original matrix, L is low-rank, and S is sparse. 
# This is what we'll be using for the background removal problem! 
# Low-rank means that the matrix has a lot of redundant information-- 
# in this case, it's the background, which is the same in every 
# scene (talk about redundant info!). Sparse means that the matrix 
# has mostly zero entries-- in this case, see how the picture of 
# the foreground (the people) is mostly empty. (In the case 
# of corrupted data, S is capturing the corrupted entries).

# https://nbviewer.org/github/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb