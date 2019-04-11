import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt

data = genfromtxt('data/Delivery.csv', delimiter=',')
x_data = data[:,:-1]
y_data = data[:,-1]

model = linear_model.LinearRegression()
model.fit(x_data,y_data)

print('Coefficient: ', model.coef_)
print('Intercept: ', model.intercept_)

x_test = [[102, 4]]
print('Prediction: ', model.predict(x_test))