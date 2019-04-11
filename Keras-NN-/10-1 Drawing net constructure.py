from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPool2D, Flatten
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("../../tensorflow/MNIST", one_hot=True)

x_train, y_train = mnist.train.images, mnist.train .labels
x_test, y_test = mnist.test.images, mnist.test.labels

# (5500,784)->(5500,28,28,1)
x_train = x_train.reshape(-1,28,28,1)
x_test = x_train.reshape(-1,28,28,1)

model = Sequential()

# input_shape only be set in first convolution layer
model.add(Convolution2D(
    input_shape=(28,28,1),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    activation='relu',
    name='conv1'
))

model.add(MaxPool2D(
    pool_size=2,
    strides=2,
    padding='same',
    name='pool1'
))

model.add(Convolution2D(64,5,strides=1,padding='same', activation='relu',name='conv2'))
model.add(MaxPool2D(2,2,'same', name='pool2'))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10,activation='softmax'))

# adam = Adam(lr=1e-4)
#
# model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(x_train, y_train, batch_size=100, epochs=10)
# loss, accuracy = model.evaluate(x_test, y_test)
#
# print('\ntest loss: ',loss)
# print('accuracy: ', accuracy)

plot_model(model, to_file='model.png', show_shapes=True, show_layer_names='False',rankdir='TB')
plt.figure(figsize=(10, 10))
img = plt.imread('model.png')
plt.imshow(img)
plt.axis('off')
plt.show()