
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
    def __init__(self, input_img, weights, biases, learning_rate):
        """
        Args:
            num_classes: The number of classes of the dataset
            
        """              
        self._input = input_img
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


# input and output vector placeholders
x = tf.placeholder(tf.float32, [None, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])


# In[4]:


# Weight parameters 
weights = {
    "wc1": tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=0.01), name="wc1"),
    "wc2": tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.01), name="wc2"),
    "wc3": tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01), name="wc3"),
    "wc4": tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.01), name="wc4"),
    "wc5": tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.01), name="wc5"),
    "wf1": tf.Variable(tf.truncated_normal([28*28*256, 4096], stddev=0.01), name="wf1"),
    "wf2": tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.01), name="wf2"),
    "wf3": tf.Variable(tf.truncated_normal([4096, n_classes], stddev=0.01), name="wf3")
}

# Bias parameters
biases = {
    "bc1": tf.Variable(tf.constant(0.0, shape=[96]), name="bc1"),
    "bc2": tf.Variable(tf.constant(1.0, shape=[256]), name="bc2"),
    "bc3": tf.Variable(tf.constant(0.0, shape=[384]), name="bc3"),
    "bc4": tf.Variable(tf.constant(1.0, shape=[384]), name="bc4"),
    "bc5": tf.Variable(tf.constant(1.0, shape=[256]), name="bc5"),
    "bf1": tf.Variable(tf.constant(1.0, shape=[4096]), name="bf1"),
    "bf2": tf.Variable(tf.constant(1.0, shape=[4096]), name="bf2"),
    "bf3": tf.Variable(tf.constant(1.0, shape=[n_classes]), name="bf3")
}

# fully connected layer
fc_layer = lambda x, W, b, name=None: tf.nn.bias_add(tf.matmul(x, W), b)

class alexNet(Model):
    # overwrite class function model_architecture
    def conv_fn(img, weights, biases, strides, padding, name): 
        conv = tf.nn.conv2d(img, weights=weights, strides=strides, padding=padding, name=name)
        conv = tf.nn.bias_add(conv, biases=biases)
        conv = tf.nn.relu(conv)            
        return conv
    
    def model_architecture(self, img, weights, biases):
        # reshape the input image vector to dimension 227 x 227 x 3
        img = tf.reshape(img, [-1, 227, 227, 3])     
        
        # 1st convolutional layer
        #conv1 = conv_fn(img, weights["wc1"], biases["bc1"], strides=[1, 4, 4, 1], padding="SAME", name="conv1")
        conv1 = tf.nn.conv2d(img, weights["wc1"], strides=[1, 4, 4, 1], padding="SAME", name="conv1")
        conv1 = tf.nn.bias_add(conv1, biases["bc1"])
        conv1 = tf.nn.relu(conv1)        
        conv1 = tf.nn.local_response_normalization(conv1, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        # 2nd convolutional layer
        #conv2 = conv_fn(img, weights["wc2"], biases["bc2"], strides=[1, 1, 1, 1], padding="SAME", name="conv2")
        conv2 = tf.nn.conv2d(conv1, weights["wc2"], strides=[1, 1, 1, 1], padding="SAME", name="conv2")
        conv2 = tf.nn.bias_add(conv2, biases["bc2"])
        conv2 = tf.nn.relu(conv2)        
        conv2 = tf.nn.local_response_normalization(conv2, depth_radius=5.0, bias=2.0, alpha=1e-4, beta=0.75)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        # 3rd convolutional layer, no max pooling        
        #conv3 = conv_fn(img, weights["wc3"], biases["bc3"], strides=[1, 1, 1, 1], padding="SAME", name="conv3")
        conv3 = tf.nn.conv2d(conv2, weights["wc3"], strides=[1, 1, 1, 1], padding="SAME", name="conv3")
        conv3 = tf.nn.bias_add(conv3, biases["bc3"])
        conv3 = tf.nn.relu(conv3)        

        # 4th convolutional layer, no max pooling
        #conv4 = conv_fn(img, weights["wc4"], biases["bc4"], strides=[1, 1, 1, 1], padding="SAME", name="conv4")
        conv4 = tf.nn.conv2d(conv3, weights["wc4"], strides=[1, 1, 1, 1], padding="SAME", name="conv4")
        conv4 = tf.nn.bias_add(conv4, biases["bc4"])
        conv4 = tf.nn.relu(conv4)        
        
        # 5th convolutional layer
        #conv5 = conv_fn(img, weights["wc5"], biases["bc5"], strides=[1, 1, 1, 1], padding="SAME", name="conv5")
        conv5 = tf.nn.conv2d(conv4, weights["wc5"], strides=[1, 1, 1, 1], padding="SAME", name="conv5")
        conv5 = tf.nn.bias_add(conv5, biases["bc5"])
        conv5 = tf.nn.relu(conv5)        
        conv5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")        
        
        # stretching out the 5th convolutional layer into a long vector
        shape = [-1, weights['wf1'].get_shape().as_list()[0]]
        flatten = tf.reshape(conv5, shape)

        # 1st fully connected layer
        fc1 = fc_layer(flatten, weights["wf1"], biases["bf1"], name="fc1")
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, keep_prob=0.5)

        # 2nd fully connected layer
        fc2 = fc_layer(fc1, weights["wf2"], biases["bf2"], name="fc2")
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, keep_prob=0.5)

        # 3rd fully connected layer
        fc3 = fc_layer(fc2, weights["wf3"], biases["bf3"], name="fc3")
        res = tf.nn.softmax(fc3)

        # Return the complete AlexNet model
        return res


# In[5]:


# Construct model
alex = alexNet(x, weights, biases, learning_rate)
pred = alex.model_architecture(x, weights, biases)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= pred, labels= y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.types.float32))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()


# In[6]:


batch_size = 64

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print ("Optimization Finished!")
    # Calculate accuracy for 256 mnist test images
    print ("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256]}))

