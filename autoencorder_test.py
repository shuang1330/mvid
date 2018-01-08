'''
TODO: transform the pandas dataframe to mnist data class form
and transforming the data using its function input_data()
'''

import pandas as pd
import os
# import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as lays
from sklearn import preprocessing
import tensorflow as tf
from math import ceil
import numpy as np
from lib.read_data import read_data_set

def autoencoder(x):
    net = lays.fully_connected(x,2)
    net = lays.fully_connected(net,180)
    # with tf.variable_scope('fc_autoencoder',regularizer=slim.l2_regularizer,
    #     initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01)):
    #     with tf.variable_scope('fc'):
    #         intermediate = slim.fully_connected(x,32,scope='fc1')
    #         net = slim.fully_connected(intermediate,x.shape[0],scope='fc2')
    return net

if __name__=='__main__':
    # input and model
    ae_inputs = tf.placeholder(tf.float32,(None, 180))
    ae_outputs = autoencoder(ae_inputs)

    # loss and training options
    loss = tf.reduce_mean((tf.square(ae_outputs-ae_inputs)))
    batch_size = 50
    epoch_num = 51
    lr = 0.001
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # initializer
    init = tf.global_variables_initializer()

    # read dataset
    all_data = pd.read_csv('data/dummy_no_nan_data.csv',sep='\t')
    mvid = read_data_set(all_data)
    batch_per_ep = ceil(mvid.train.num_examples/batch_size)

    # start training
    with tf.Session() as sess:
        sess.run(init)

        for ep in range(epoch_num):
            for batch_no in range(batch_per_ep):
                batch_data, batch_label = mvid.train.next_batch(batch_size)
                recon_data,error = sess.run([ae_outputs,loss],
                feed_dict={ae_inputs:batch_data})
                print('Epoch: {0}\tIteration:{1}\tError: {2}\t'.format(
                ep, batch_no, error
                ))

        # test the trained network
        batch_data, batch_label = mvid.test.next_batch(batch_size=50)
        recon_data = sess.run([ae_outputs,loss],
        feed_dict={ae_inputs:batch_data})
        print('Test dataset\tError: {0}'.format(error))
