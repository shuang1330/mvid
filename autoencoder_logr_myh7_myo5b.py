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
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

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
    epoch_num = 50
    lr = 0.001

    # read dataset

    all_data = pd.read_csv('data/all_variants/myh7_myo5b_dummy_no_nan.csv',sep='\t')
    # all_data = pd.read_csv('/Users/gcc/projects/myo5b_project/data/dummy_no_nan_data.csv',sep='\t')
    # all_data = pd.read_csv('data/myh7/mhy7_dummy_no_nan.tsv',sep='\t')
    # print(all_data.shape)
    # print(all_data.head())
    # raise NotImplementedError


    myo5b_with_genename = all_data.loc[all_data['genename']=='MYO5B']
    myh7_with_genename = all_data.loc[all_data['genename']=='MYH7']

    myo5b_with_genename = myo5b_with_genename.drop(['genename'],axis=1)
    myh7_with_genename = myh7_with_genename.drop(['genename'],axis=1)

    myh7 = read_data_set(myh7_with_genename)
    myo5b = read_data_set(myo5b_with_genename,test_size=0)
    batch_per_ep = ceil(myh7.train.num_examples/batch_size)

    # ======================== autoencoder ==============================
    # model
    inputs = tf.placeholder(tf.float32,(None, myh7.train.num_features))
    labels = tf.placeholder(tf.float32,(None, 2))
    ae_outputs,ae_bottle = autoencoder(inputs)
    # loss and training options
    loss_ae = tf.reduce_mean((tf.square(ae_outputs-inputs))) # autoencoder
    # loss_fn = tf.reduce_mean((tf.square(fn_outputs-labels)))
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_ae)

    # initializer
    init = tf.global_variables_initializer()

    # start training
    with tf.Session() as sess:
        sess.run(init)
        for ep in range(epoch_num):
            for batch_no in range(batch_per_ep):
                batch_data, batch_label = myh7.train.next_batch(batch_size)
                batch_label_onehot = dense_to_one_hot(batch_label,2)
                _, recon_data,error = sess.run([train_op,
                                                ae_outputs,loss_ae],
                                                feed_dict={inputs:batch_data,
                                                labels:batch_label_onehot})
                print('Epoch: {0}\tIteration:{1}\tError: {2}\t'.format(
                ep, batch_no, error
                ))

        # test the trained network
        batch_data, batch_label = myh7.test.next_batch(batch_size=50)
        batch_label_onehot = dense_to_one_hot(batch_label,2)
        recon_data, bottle, error = sess.run([ae_outputs,ae_bottle, loss_ae],
        feed_dict={inputs:batch_data,labels:batch_label_onehot})
        print('Test dataset\tError: {0}'.format(error))

        # extract training features from the ae
        batch_label_onehot = dense_to_one_hot(myh7.train.labels,2)
        _,train_ae_features,_ = sess.run([ae_outputs,ae_bottle, loss_ae],
        feed_dict={inputs:myh7.train.values,labels:batch_label_onehot})
        batch_label_onehot = dense_to_one_hot(myh7.test.labels,2)
        _,test_ae_features,_ = sess.run([ae_outputs,ae_bottle, loss_ae],
                                    feed_dict={
                                    inputs:myh7.test.values,
                                    labels:batch_label_onehot
                                    })
        classifier_logr = LogisticRegression(class_weight='balanced')
        # print(train_ae_features.shape, myh7.test.labels.shape)
        classifier_logr.fit(train_ae_features,myh7.train.labels)
        pred = classifier_logr.predict(test_ae_features)
        tn, fp, fn, tp = confusion_matrix(myh7.test.labels,pred).ravel()
        sensitivity = tp/(fn+tp)
        specificity = tn/(fp+tn)
        print(tn,fp,fn,tp)
        print(sensitivity,specificity)

        # test on myo5b dataset
        batch_label_onehot = dense_to_one_hot(myo5b.train.labels,2)
        _,train_ae_features,_ = sess.run([ae_outputs,ae_bottle, loss_ae],
        feed_dict={inputs:myo5b.train.values,labels:batch_label_onehot})
        pred = classifier_logr.predict(train_ae_features)
        tn, fp, fn, tp = confusion_matrix(myo5b.train.labels,pred).ravel()
        sensitivity = tp/(fn+tp)
        specificity = tn/(fp+tn)
        print(tn,fp,fn,tp)
        print(sensitivity,specificity)
        # use bottleneck as new features for logistic regression

    #
    # # ======================= feedfoward networks ==============================
    # # model
    # inputs = tf.placeholder(tf.float32,(None, 991))
    # labels = tf.placeholder(tf.float32,(None, 2))
    # # ae_outputs = autoencoder(inputs)
    # fn_outputs,fn_probs = feedforward_net(inputs)
    # # loss and training options
    # # loss_ae = tf.reduce_mean((tf.square(ae_outputs-inputs))) # autoencoder
    # loss_fn = tf.reduce_mean((tf.square(fn_outputs-labels)))
    # train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_fn)
    #
    # # initializer
    # init = tf.global_variables_initializer()
    #
    # # start training
    # mvid.train._epochs_completed = 0
    # with tf.Session() as sess:
    #     sess.run(init)
    #     for ep in range(epoch_num):
    #         mvid.train._epochs_completed += 1
    #         for batch_no in range(batch_per_ep):
    #             mvid.train._index_in_epoch = batch_no
    #             mvid.train._epochs_completed += 1
    #             batch_data, batch_label = mvid.train.next_batch(batch_size)
    #             batch_label_onehot = dense_to_one_hot(batch_label,2)
    #             _, recon_data,error,probs = sess.run([train_op,
    #                                             fn_outputs,
    #                                             loss_fn,
    #                                             fn_probs],
    #                                             feed_dict={inputs:batch_data,
    #                                             labels:batch_label_onehot})
    #             print('Epoch: {0}\tIteration:{1}\tError: {2}\t'.format(
    #             ep, batch_no, error
    #             ))
    #
    #     # test the trained network
    #     batch_data, batch_label = mvid.test.values, mvid.test.labels
    #     # print(batch_data.shape, batch_label.shape)
    #     batch_label_onehot = dense_to_one_hot(batch_label,2)
    #     recon_data,error, probs = sess.run([fn_outputs,loss_fn,fn_probs],
    #                           feed_dict={inputs:batch_data,
    #                           labels:batch_label_onehot})
    #
    #     print(batch_label_onehot[:,0])
    #     probs = probs[:,0]
    #     probs[probs>0.5] = 1
    #     probs[probs<0.5] = 0
    #     print(probs)
    #     tn, fp, fn, tp = confusion_matrix(batch_label_onehot[:,0],
    #                                       probs).ravel()
    #     print('Test dataset\tError: {0}'.format(error))
    #     print(tn, fp, fn, tp)
