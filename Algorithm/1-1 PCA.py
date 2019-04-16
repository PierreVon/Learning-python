import numpy as np
from sklearn.datasets import load_iris

m = 30
n = 1
iris = load_iris().data[:30,:]
data = iris[:, 1:]

C = np.matmul(data.T, data) / m
print(C)
eigenValues, eigenVectors = np.linalg.eig(C)
print('EigenValues: ', eigenValues, '\nEigenVectors:\n', eigenVectors)

print(np.matmul(eigenVectors[:, : n].T, data.T))