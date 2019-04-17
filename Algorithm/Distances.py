import numpy as np
from sklearn.datasets import load_iris

data = load_iris().data[:1, :3]


# data [[X1], [X2], ..., [Xn]]
def Mahalanobis(data):
    if len(data.T) == 1:
        return 0
    else:
        cov = np.cov(data)
        cov = np.mat(cov)
        u = np.mat(np.mean(data, axis=1))
        c = data - u.T
        cx, cy = c.shape
        if cx == cy and np.linalg.det(c) == 0:
            exit('singular matrix')
        elif np.linalg.det(cov) == 0:
            exit('covariance is a singular matrix')
        else:
            mah = np.matmul(np.matmul(c.T, cov.I), c)
            return mah


def Manhattan(data):
    cols, rows = data.shape
    dist = np.zeros((rows, rows), np.float32)
    for i in range(rows):
        for j in range(rows):
            dist[i, j] = np.sum(data[:, i] - data[:, j])
    return dist


def Euclidean(data):
    cols, rows = data.shape
    dist = np.zeros((rows, rows), np.float32)
    for i in range(rows):
        for j in range(rows):
            dist[i, j] = np.sqrt(np.sum(np.power((data[:, i] - data[:, j]), 2)))
    return dist


def Minkowski(data, q):
    cols, rows = data.shape
    dist = np.zeros((rows, rows), np.float32)
    for i in range(rows):
        for j in range(rows):
            dist[i, j] = np.sqrt(np.sum(np.power((data[:, i] - data[:, j]), q)))
    return dist

def Chebyshev(data):
    cols, rows = data.shape
    dist = np.zeros((rows, rows), np.float32)
    for i in range(rows):
        for j in range(rows):
            dist[i, j] = np.max(data[:, i] - data[:, j])
    return dist


def Cosine(data):
    cols, rows = data.shape
    dist = np.zeros((rows, rows), np.float32)
    for i in range(rows):
        for j in range(rows):
            dist[i, j] = np.dot(data[:, i], data[:, j])/(np.linalg.norm(data[:, i]) * np.linalg.norm(data[:, j]))
    return dist


#print(Mahalanobis(data.T))
#print(Manhattan(data.T))
#print(Euclidean(data.T))
#print(Minkowski(data.T, 4))
#print(Chebyshev(data.T))
print(Cosine(data.T))


