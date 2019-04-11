from tensorflow.examples.tutorials.mnist import input_data
from keras.models import load_model

mnist = input_data.read_data_sets("../../tensorflow/MNIST", one_hot=True)

x_train, y_train = mnist.train.images, mnist.train .labels
x_test, y_test = mnist.test.images, mnist.test.labels

model = load_model('model.h5')
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss: ',loss)
print('accuracy: ', accuracy)

# it can keeps training
model.fit(x_train,y_train,batch_size=64,epochs=4)

loss, accuracy = model.evaluate(x_test, y_test)
print('\ntest loss: ',loss)
print('accuracy: ', accuracy)