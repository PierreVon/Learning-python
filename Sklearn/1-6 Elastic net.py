from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

iris = load_iris().data[:30,:]
x_data = iris[:,1:]
y_data = iris[:,0]

# combine Ridge and LASSO
# lambda * cumsum (alpha * theta^2 + ( 1 - alpha ) | theta | )
# it also sets coefficient as 0 to eliminate some correlated variables
model = linear_model.ElasticNetCV()
model.fit(x_data,y_data)

print(model.coef_)
print(model.alpha_)

i = 0
print('prediction value is: ', model.predict(x_data[i, np.newaxis]))
print('original value is: ', y_data[i])
