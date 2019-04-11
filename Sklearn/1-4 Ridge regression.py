from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

iris = load_iris().data[:30,:]
x_data = iris[:,1:]
y_data = iris[:,0]

# default generation is 50
alphas = np.linspace(0.001, 1)
# fitting 50 parameters to find the best one
# lambda * cumsum (alpha * theta ^2 )
model = linear_model.RidgeCV(alphas=alphas)
model.fit(x_data, y_data)

# best ridge parameter
print(model.alpha_)

i = 0
print('prediction value is: ', model.predict(x_data[i, np.newaxis]))
print('original value is: ', y_data[i])
