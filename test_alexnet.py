# coding: utf-8

# # Python Object Oriented Programming
# Date: 2018/8/31
# Author: Alex Hsu
# Purpose: take deep learning model AlexNet as an example, create a class of DNN Model
# 
# DataSet: MNIST
# number of classes: 10
# 
# References:
# * Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. "ImageNet Classification with Deep Convolutional Neural Networks". NIPS, 2012.
# * [rahulbhalley/AlexNet-TensorFlow](https://github.com/rahulbhalley/AlexNet-TensorFlow)

# In[1]:


import tensorflow as tf

print(tf.__version__)


# In[2]:


class Model(object):
    """Base model for building the AlexNet model"""
    def __init__(self, img, weights, biases, learning_rate):
        """
        Args:
            num_classes: The number of classes of the dataset
            
        """              
        self._input = img
        self._weights = weights
        self._biases = biases
        
        if learning_rate:
            self.learning_rate = learning_rate  
        else: 
            self.learning_rate = 0.1
            
    def model_architecture(self, img, weights, biases):        
        return img
        
    def train(self, mode='train'):
        return 
    def inferance(self, mode='inference'):        
        return 


# In[3]:


#mnist = mnist.read_data_sets('.', one_hot=True)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Network Parameters
n_inputs = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
learning_rate = 0.001
dropout = 0.8 # Dropout, probability to keep units

# input and output vector placeholders
x = tf.placeholder(tf.float32, [None, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# In[4]:


# Weight parameters 
weights = {
    "wc1": tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=0.01), name="wc1"),
    "wc2": tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.01), name="wc2"),
    "wc3": tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.01), name="wc3"),
    # "wc4": tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01), name="wc4"),
    # "wc5": tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01), name="wc5"),
    "wf1": tf.Variable(tf.truncated_normal([4*4*256, 1024], stddev=0.01), name="wf1"),
    "wf2": tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.01), name="wf2"),
    "wf3": tf.Variable(tf.truncated_normal([1024, n_classes], stddev=0.01), name="wf3")
}

# Bias parameters
biases = {
    "bc1": tf.Variable(tf.constant(0.0, shape=[64]), name="bc1"),
    "bc2": tf.Variable(tf.constant(1.0, shape=[128]), name="bc2"),
    "bc3": tf.Variable(tf.constant(0.0, shape=[256]), name="bc3"),
    # "bc4": tf.Variable(tf.constant(1.0, shape=[384]), name="bc4"),
    # "bc5": tf.Variable(tf.constant(1.0, shape=[256]), name="bc5"),
    "bf1": tf.Variable(tf.constant(1.0, shape=[1024]), name="bf1"),
    "bf2": tf.Variable(tf.constant(1.0, shape=[1024]), name="bf2"),
    "bf3": tf.Variable(tf.constant(1.0, shape=[n_classes]), name="bf3")
}

# fully connected layer
fc_layer = lambda x, W, b, name=None: tf.nn.bias_add(tf.matmul(x, W), b)

class alexNet(Model):
    def conv2d(self, name, l_input, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

    def max_pool(self, name, l_input, k):
        return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

    def norm(self, name, l_input, lsize=4):
        return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

    def model_architecture(self, img=None, weights=None, biases=None, dropout=None):
        if img is None:
            img = self._input # use init weights
        if weights is None:
            weights = self._weights # use init weights
        if biases is None:
            biases = self._biases # use init biases

        # Reshape input picture
        _X = tf.reshape(img, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = self.conv2d('conv1', _X, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        pool1 = self.max_pool('pool1', conv1, k=2)
        # Apply Normalization
        norm1 = self.norm('norm1', pool1, lsize=4)
        # Apply Dropout
        norm1 = tf.nn.dropout(norm1, dropout)

        # Convolution Layer
        conv2 = self.conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        pool2 = self.max_pool('pool2', conv2, k=2)
        # Apply Normalization
        norm2 = self.norm('norm2', pool2, lsize=4)
        # Apply Dropout
        norm2 = tf.nn.dropout(norm2, dropout)

        # Convolution Layer
        conv3 = self.conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
        # Max Pooling (down-sampling)
        pool3 = self.max_pool('pool3', conv3, k=2)
        # Apply Normalization
        norm3 = self.norm('norm3', pool3, lsize=4)
        # Apply Dropout
        norm3 = tf.nn.dropout(norm3, dropout)

        # Fully connected layer
        dense1 = tf.reshape(norm3, [-1, weights['wf1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
        dense1 = tf.nn.relu(tf.matmul(dense1, weights['wf1']) + biases['bf1'], name='fc1') # Relu activation

        dense2 = tf.nn.relu(tf.matmul(dense1, weights['wf2']) + biases['bf2'], name='fc2') # Relu activation

        # Output, class prediction
        out = tf.matmul(dense2, weights['wf3']) + biases['bf3']
        return out        


# In[5]:


# Construct model
alex = alexNet(x, weights, biases, learning_rate) # create an alexNet object
pred = alex.model_architecture(dropout = keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= pred, labels= y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.types.float32))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


# In[6]:

display_step = 20
batch_size = 64
training_iters = 20000

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print ("Optimization Finished!")
    # Calculate accuracy for 256 mnist test images
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.}))
