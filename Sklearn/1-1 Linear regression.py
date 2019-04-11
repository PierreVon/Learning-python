from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.rand(100)
noize = np.random.normal(0,0.03,100)
y_data = x_data * 0.1 + noize

# change (100,) -> (100,1)
x_data = x_data[:, np.newaxis]
y_data = y_data[:, np.newaxis]

model = LinearRegression()
model.fit(x_data,y_data)

plt.scatter(x_data,y_data)
plt.plot(x_data, model.predict(x_data), 'r-')
plt.show()
