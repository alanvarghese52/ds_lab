import numpy as np
mat1=np.array([[12,23,22],[5,87,34],[44,77,3]])
mat2=np.array([[12,32,22],[7,78,43],[44,77,3]])
print('ADDITION')
print(np.add(mat1,mat2))
print('SUBTRACTION')
print(np.subtract(mat1,mat2))
print('DIVISION')
print(np.divide(mat1,mat2))
print('MULTIPLY')
print(np.multiply(mat1,mat2))
print("-----------------------------")
from numpy import array
from scipy.linalg import svd
A = array([[12, 21, 39], [94, 75, 46], [37, 80, 94], [64, 34, 99], [38, 12, 89]])
u, s, vt = svd(A)
print('Decomposed matrix:\n ', u)
print('Inverse matrix:\n ', s)
print('Transpose matrix:\n ', vt)
