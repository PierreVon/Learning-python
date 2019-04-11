from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import SimpleRNN
from keras.optimizers import Adam

input_size = 28
time_steps = 28
# hidden layer, number of cell
cell_size = 50

mnist = input_data.read_data_sets("../../tensorflow/MNIST", one_hot=True)

x_train, y_train = mnist.train.images, mnist.train .labels
x_test, y_test = mnist.test.images, mnist.test.labels

# (55000,784)->(55000,28,28)
x_train = x_train.reshape(-1,28,28)
x_test = x_train.reshape(-1,28,28)

model = Sequential()

model.add(SimpleRNN(
    units=cell_size,
    input_shape=(time_steps, input_size)
))

model.add(Dense(10, activation='softmax'))
adam = Adam(1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=10)
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss: ',loss)
print('accuracy: ', accuracy)