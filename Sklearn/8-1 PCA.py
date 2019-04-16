from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

iris = load_iris().data[:30,:]
x_data = iris[:, 1:]
pca = PCA(n_components=2)
pca.fit(x_data)
print(pca.explained_variance_ratio_)