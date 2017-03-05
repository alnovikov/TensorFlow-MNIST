'''
This code is a walkthrough from official TensorFlow tutorials.
MNIST with softmax regression aka as 'hello world' in TensorFlow =)

run virtual environment first in bash:
    source ~/tensorflow/bin/activate

NB: use 'deactivate' to close venv

'''
#install dependencies
import tensorflow as tf
import numpy as np

#load in MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#define the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# add a new placeholder to input the correct answers:
y_ = tf.placeholder(tf.float32, [None, 10])

#implement the cross-entropy function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

#initialize Gradient Descent with a step of 0.3
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)

#launch TensorFlow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#batch implementation aka 'stochastic training'
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#get predictions
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#accuracy on test data
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
