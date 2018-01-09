'''
two fc layers for autoencoder
'''

import pandas as pd
import os
# import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as lays
from sklearn import preprocessing
import tensorflow as tf
from math import ceil
import numpy as np
from lib.read_data import dataset,Datasets
from lib.net import autoencoder,feedforward_net
from sklearn.model_selection import train_test_split


def dense_to_one_hot(labels_dense, num_classes):
  """
  Convert class labels from scalars to one-hot vectors.
  """
  # num_labels = labels_dense.shape[0]
  # index_offset = np.arange(num_labels) * num_classes
  # labels_one_hot = np.zeros((num_labels, num_classes))
  # labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return np.eye(num_classes)[np.array(labels_dense).reshape(-1)]

def read_data_set(data_table,test_size=0.25):
    '''
    convert a pandas dataframe data table into Datasets(dataset,dataset)
    '''
    train, test = train_test_split(data_table,test_size=0.25)
    train_x = np.array(train[[col for col in train.columns
    if col not in ['INFO']]])
    test_x = np.array(test[[col for col in train.columns
    if col not in ['INFO']]])
    train_y = np.array(train['INFO'],dtype=np.int8)
    test_y = np.array(test['INFO'],dtype=np.int8)
    return Datasets(train=dataset(train_x,train_y),
    test=dataset(test_x,test_y))

if __name__=='__main__':
    # constant
    batch_size = 50
    epoch_num = 51
    lr = 0.001

    # read dataset
    all_data = pd.read_csv('data/dummy_no_nan_data.csv',sep='\t')
    mvid = read_data_set(all_data)
    batch_per_ep = ceil(mvid.train.num_examples/batch_size)

    # model
    inputs = tf.placeholder(tf.float32,(None, 180))
    labels = tf.placeholder(tf.float32,(None, 2))
    ae_outputs = autoencoder(inputs)
    fn_outputs = feedforward_net(inputs)

    # loss and training options
    loss_ae = tf.reduce_mean((tf.square(ae_outputs-inputs))) # autoencoder
    loss_fn = tf.reduce_mean((tf.square(fn_outputs-labels)))
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_fn)

    # initializer
    init = tf.global_variables_initializer()

    # start training
    with tf.Session() as sess:
        sess.run(init)

        for ep in range(epoch_num):
            for batch_no in range(batch_per_ep):
                batch_data, batch_label = mvid.train.next_batch(batch_size)
                batch_label_onehot = dense_to_one_hot(batch_label,2)
                recon_data,error = sess.run([ae_outputs,loss_fn],
                feed_dict={inputs:batch_data,labels:batch_label_onehot})
                print('Epoch: {0}\tIteration:{1}\tError: {2}\t'.format(
                ep, batch_no, error
                ))

        # test the trained network
        batch_data, batch_label = mvid.test.next_batch(batch_size=50)
        recon_data = sess.run([ae_outputs,loss_fn],
        feed_dict={inputs:batch_data,labels:batch_label_onehot})
        print('Test dataset\tError: {0}'.format(error))
