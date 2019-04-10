import keras
import numpy as np
import matplotlib.pyplot as plt
# construct model order by sequence
from keras.models import Sequential
# fully connected
from keras.layers import Dense

x_data = np.random.rand(100)
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = x_data*0.1 + 0.2 + noise

# modeling
# construct a sequential model
model = Sequential()
# add a fully connected layer
model.add(Dense(units=1, input_dim=1))
model.compile(optimizer='sgd', loss='mse')

for step in range(2001):
    cost = model.train_on_batch(x_data, y_data)
    if step % 500 == 0:
        print("cost: ",cost)

W, b = model.layers[0].get_weights()
print('W:',W,'b:',b)

y_pred = model.predict(x_data)

plt.scatter(x_data,y_data)
plt.plot(x_data, y_pred, 'r-', lw=3)
plt.show()