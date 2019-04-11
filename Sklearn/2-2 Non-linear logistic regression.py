import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.datasets import make_gaussian_quantiles
from sklearn.preprocessing import  PolynomialFeatures
from sklearn import linear_model

x_data, y_data = make_gaussian_quantiles(n_samples=500, n_features=2, n_classes=2)

logistic = linear_model.LogisticRegression()
logistic.fit(x_data, y_data)

x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
y_min, y_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# a = [1, 2], b = [3, 4]
# np.r_(a, b) [1, 2, 3, 4]
# np.c_(a, b) [[1, 3], [2, 4]]
# zip into one dimension, ravel return view, flatten return copy
z = logistic.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

cs = plt.contour(xx, yy, z)
# fill with color
# cs = plt.contourf(xx, yy, z)=
print('Linear logistic regression score: ',logistic.score(x_data, y_data))

##################################
# Non-linear logistic regression #
##################################

poly = PolynomialFeatures(degree=5)
x_poly = poly.fit_transform(x_data)
logistic = linear_model.LogisticRegression()
logistic.fit(x_poly, y_data)


x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
y_min, y_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
z = logistic.predict(poly.fit_transform(np.c_[xx.ravel(), yy.ravel()]))
z = z.reshape(xx.shape)
cs = plt.contour(xx, yy, z)
print('Non-linear logistic regression score: ',logistic.score(x_poly, y_data))

# simple way
plt.scatter(x_data[:,0], x_data[:,1], c = y_data)
plt.show()