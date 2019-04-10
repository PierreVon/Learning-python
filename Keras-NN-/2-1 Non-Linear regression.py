import keras
import numpy as np
import matplotlib.pyplot as plt
# construct model order by sequence
from keras.models import Sequential
# fully connected
from keras.layers import Dense,Activation
from keras.optimizers import SGD

x_data = np.linspace(-0.5, 0.5, 200)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# modeling
# construct a sequential model
model = Sequential()
# add a
# add two fully connected layer, 1-10-1
model.add(Dense(units=10, input_dim=1, activation='relu'))
#model.add(Activation('tanh'))
model.add(Dense(units=1))
model.add(Activation('tanh'))

# change the default learning rate
sgd = SGD(lr=0.3)
model.compile(optimizer=sgd, loss='mse')

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