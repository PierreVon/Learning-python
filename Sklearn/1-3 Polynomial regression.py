import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = np.genfromtxt('data/Job.csv', delimiter=',')
x_data = data[1:,0,np.newaxis]
y_data = data[1:,1,np.newaxis]


model = LinearRegression()
model.fit(x_data,y_data)

# degree can adjust feature of polynomial
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x_data)
print(x_poly)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y_data)

plt.scatter(x_data,y_data)
plt.plot(x_data, lin_reg.predict(poly_reg.fit_transform(x_data)),'b')
plt.plot(x_data, model.predict(x_data), 'r-')
plt.show()