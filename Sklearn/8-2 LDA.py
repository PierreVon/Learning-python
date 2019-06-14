from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = load_iris()
x_data = iris.data
y = iris.target

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(x_data, y)
print(lda.explained_variance_ratio_)