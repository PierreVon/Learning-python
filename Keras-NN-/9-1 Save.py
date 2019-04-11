from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

mnist = input_data.read_data_sets("../../tensorflow/MNIST", one_hot=True)

x_train, y_train = mnist.train.images, mnist.train .labels
x_test, y_test = mnist.test.images, mnist.test.labels

print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Dense(units=10, input_dim=784, bias_initializer='one', activation='tanh'))
sgd = SGD(lr=0.2)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=10)
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss: ',loss)
print('accuracy: ', accuracy)

model.save('model.h5')

# only save parameters
# model.save_weights('weights.h5')

# only save constructure
# from keras.models import model_from_json
# json_string = model.to_json()
# model = model_from_json(json_string)
