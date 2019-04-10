from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam

mnist = input_data.read_data_sets("../../tensorflow/MNIST", one_hot=True)

x_train, y_train = mnist.train.images, mnist.train .labels
x_test, y_test = mnist.test.images, mnist.test.labels

print(x_train.shape)
print(y_train.shape)

model = Sequential([
    Dense(units=10, input_dim=784, bias_initializer='one', activation='softmax')
])
sgd = SGD(lr=0.2)
adam = Adam(lr=0.01)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=10)
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss: ',loss)
print('accuracy: ', accuracy)