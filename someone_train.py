# -*- coding: utf-8 -*-
from __future__ import print_function
import random
import numpy as np
import tensorflow as tf

from data_input import extract_data, resize_with_pad, IMAGE_SIZE

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 10

# Network Parameters
n_classes = 2  # total classes
dropout = 0.75 # Dropout, probability to keep units


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# tf Graph input
x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
def conv_net_model(x, weights, biases, dropout):
    # Reshape input picture
    # x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([16*16*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def train(report_steps):
    # Construct model
    pred = conv_net_model(x, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()
    # saver init
    saver = tf.train.Saver(tf.all_variables())
    
    with tf.Session() as sess:
        sess.run(init)
        # load checkpoint
        checkpoint = tf.train.get_checkpoint_state("saved_networks")

        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print ('Successfully loaded %s '  % (checkpoint.model_checkpoint_path))
        else:
            print ('Could not find old network weights')

        images, labels = extract_data('./data/')
        labels = np.reshape(labels, [-1 , 2])

        test_batch_x , test_batch_y = batch_data(batch_size, images, labels) 

        # print (test_batch_x.shape)
        # print (test_batch_y.shape)
        batch_idx = 0
        batch_iter = 0
        try:
            for batch_idx in xrange(training_iters):
                last_batch_idx = 0
                train_batch_x , train_batch_y = batch_data(batch_size, images, labels) 
                # print (train_batch_x.shape)
                # print (train_batch_y)

                sess.run(optimizer, feed_dict={x: train_batch_x, y: train_batch_y, keep_prob: dropout})

                if batch_idx % report_steps == 0:
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: train_batch_x, y: train_batch_y, keep_prob: 1.})
                    print (str(batch_iter) +" : "+ str(loss) + "  --  "+  str(acc))
                batch_iter = batch_iter +1

        except KeyboardInterrupt:
            print ("save weights")
            # save progress
            saver.save(sess, 'saved_networks/'+'someone', global_step = batch_idx)

def batch_data(batch_size, images, labels):
    total_batch, _ = labels.shape
    data_range = random.randint(0, total_batch)
    rn = random.randint(0, total_batch)
    batch_x = images[rn : rn+1 , :, :, : ]
    batch_y = labels[rn : rn +1, : ]
    # batch_x = images[data_range - batch_size : data_range , :, :, : ]
    # batch_y = labels[data_range - batch_size : data_range , : ]
    for i in range(batch_size):
        rn = random.randint(0, total_batch)
        x = images[rn : rn +1, :, :, : ]
        y = labels[rn : rn+1 , : ]
        batch_x = np.append(batch_x, x , axis = 0)
        batch_y = np.append(batch_y, y , axis = 0)

    # # print (labels.shape)
    # batch_x = images[data_range - batch_size : data_range/2 , :, :, : ]
    # batch_y = labels[data_range - batch_size : data_range/2 , : ]

    # data_range = random.randint(batch_size, total_batch)
    # batch_a = images[data_range - batch_size : data_range/2 , :, :, : ]
    # batch_b = labels[data_range - batch_size : data_range/2 , : ]
    # # print (labels.shape)
    # batch_x = np.append( batch_a, batch_x)
    # batch_y = np.append( batch_b, batch_y )

    # print (batch_y.shape)
    return batch_x, batch_y

if __name__ == '__main__':

    train(report_steps=20)