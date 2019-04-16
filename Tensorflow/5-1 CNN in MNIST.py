import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST", one_hot=True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    # x [batch, in_height, in_width, in_channels]
    # filter / kernel [filter_height, filter_width, in_channels, out_channels]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    # ksize = [1, x, y, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# change format of x
x_image = tf.reshape(x, [-1, 28, 28, 1])

# initialize first convolution layer's weights and biases
# 5*5 simple window, extracting features from 1 plane by 4 kernels
# third 1 is 1-channel
W_conv1 = weight_variable([5, 5, 1, 4])
# every bias to every kernel
b_conv1 = bias_variable([4])

# convolution applied ReLU function
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# initialize second convolution layer's weights and biases
# 5*5 simple window, extracting features from 4 planes by 8 kernels
W_conv2 = weight_variable([5, 5, 4, 8])
b_conv2 = bias_variable([8])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# original picture' size is 28*28, after first convolution becomes to 28*28, after first pooling changes to 14*14
# second convolution changes to 14*14, second pooling changes to 7*7
# now we get 8 planes of 7*7

# initialize first fully connected layer
# former layer has 7*7*64 neurons
W_fc1 = weight_variable([7*7*8, 100])
b_fc1 = bias_variable([100])

# flatten second pooling layer into one dimension
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*8])
# first output of first fully connected layer
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# control neuron output probability
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# second fully connected layer
W_fc2 = weight_variable([100, 10])
b_fc2 = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:0.7})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob:1.0})
        print("Iter " + str(epoch) + ", Testing Accuracy " + str(acc))



