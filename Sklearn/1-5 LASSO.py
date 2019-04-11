from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

iris = load_iris().data[:30,:]
x_data = iris[:,1:]
y_data = iris[:,0]

# the least absolute shrinkage and selectionator operator
# it automatically runs 100 default parameters to find best one
# lambda * cumsum (alpha * |theta|  )
model = linear_model.LassoCV()
model.fit(x_data,y_data)

# it sets coefficient as 0 to eliminate some correlated variables
# while Ridge hardly do such thing
print(model.coef_)
print(model.alpha_)

i = 0
print('prediction value is: ', model.predict(x_data[i, np.newaxis]))
print('original value is: ', y_data[i])