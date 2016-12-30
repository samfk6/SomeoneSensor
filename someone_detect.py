# -*- coding:utf-8 -*-
#!/usr/bin/python
import cv2
import time
import random
import cv2
import numpy as np
import tensorflow as tf
from image_show import show_image
from data_input import extract_data, resize_with_pad, IMAGE_SIZE


# Network Parameters
n_classes = 2  # total classes

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

    return batch_x, batch_y


def detect():

    # Construct model
    pred = conv_net_model(x, weights, biases, keep_prob)
    ans = tf.argmax(pred, 1)
    ans_co = tf.argmax(y, 1)

    # Initializing the variables
    init = tf.global_variables_initializer()
    # saver init
    saver = tf.train.Saver(tf.global_variables())
    # load checkpoint 
    with tf.Session() as sess:

        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print ('Successfully loaded %s '  % (checkpoint.model_checkpoint_path))
        else:
            print ('Could not find old network weights')

        # images, labels = extract_data('./data/')
        # test_batch_x , test_batch_y = batch_data(30, images, labels) 
        # result ,result2= sess.run([ans,ans_co], feed_dict = {x: test_batch_x, y: test_batch_y , keep_prob: 1} )
        tmp = 0
        while True:
            ret, frame = capture.read()
            # frame = cv2.resize(frame,(320, 240))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # a = cv2.resize(frame,(64, 64))
            # a = cv2.resize(frame,(64, 64))
            # a = np.reshape(a, [-1 , 64, 64, 3])
            # a = a /255.

            face = face_cascade.detectMultiScale(gray,1.1,5)

            # img = cv2.imread("./data/boss/2.bmp")
            # img = cv2.resize(img,(64, 64))
            # # print img.shape
            # img = np.reshape(img, [-1 , 64, 64, 3])
            # img = img /255.

            for (x1,y1,w,h) in face:

                roi = frame[y1:y1+h, x1:x1+w]
                cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(255,0,0),2)
                # cv2.imwrite( TARGET_DIR + str(tmp)+".bmp", roi )
                roi = cv2.resize(roi,(64, 64))
                roi = np.reshape(roi, [1 , 64, 64, 3])
                roi = roi /255.

                result = sess.run(ans, feed_dict = {x: roi,  keep_prob: 1} )
                print result[0]

                pre_tmp = result[0]
                if result[0] == 1:  # boss
                    tmp = tmp + 1

                    if tmp > 2:
                            print('Boss is approaching')
                            tmp = 0
                            show_image()
                else:
                    print('Not boss')
                    tmp = 0
                
                print ("TMP:" + str(tmp))



            cv2.imshow('frame',frame)
            
            if cv2.waitKey(10) == 27:
                break
            
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    detect()

    # 
    # ""
    # img = cv2.imread("./data/other/2.bmp")
    # img = cv2.resize(img,(64, 64))
    # # print img.shape
    # img = np.reshape(img, [-1 , 64, 64, 3])
    # img = img /255.
    # print img.shape
    # detect(img)
# ""