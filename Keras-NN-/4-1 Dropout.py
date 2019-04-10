from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

mnist = input_data.read_data_sets("../../tensorflow/MNIST", one_hot=True)

x_train, y_train = mnist.train.images, mnist.train .labels
x_test, y_test = mnist.test.images, mnist.test.labels

print(x_train.shape)
print(y_train.shape)

dropuot = 0.4

model = Sequential([
    Dense(units=200, input_dim=784, bias_initializer='one', activation='tanh'),
    Dropout(dropuot),
    Dense(units=100, input_dim=784, bias_initializer='one', activation='tanh'),
    Dropout(dropuot),
    Dense(units=10, input_dim=784, bias_initializer='one', activation='softmax')
])
sgd = SGD(lr=0.2)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=10)
loss, accuracy = model.evaluate(x_test, y_test)
loss_t, accuracy_t = model.evaluate(x_train, y_train)

print('\ntest loss: ',loss)
print('test accuracy: ', accuracy)
print('\nloss: ',loss_t)
print('accuracy: ', accuracy_t)