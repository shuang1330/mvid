'''
autoencoder
'''

import pandas as pd
import os
# import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as lays
import tensorflow as tf
from math import ceil
import numpy as np
from lib.read_data import dataset,Datasets
from lib.net import autoencoder,feedforward_net
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier

def dense_to_one_hot(labels_dense, num_classes):
  '''
  Convert class labels from scalars to one-hot vectors.
  '''
  return np.eye(num_classes)[np.array(labels_dense).reshape(-1)]

def read_data_set(data_table,test_size=0.25):
    '''
    convert a pandas dataframe data table into Datasets(dataset,dataset)
    '''
    train=data_table.sample(frac=(1-test_size),random_state=200)
    test=data_table.drop(train.index)
    # train, test = train_test_split(data_table,test_size=0.25)
    train_x = np.array(train[[col for col in train.columns
    if col not in ['INFO']]])
    test_x = np.array(test[[col for col in train.columns
    if col not in ['INFO']]])
    train_y = np.array(train['INFO'],dtype=np.int8)
    test_y = np.array(test['INFO'],dtype=np.int8)
    return Datasets(train=dataset(train_x,train_y),
    test=dataset(test_x,test_y))

if __name__=='__main__':
    # read dataset
    all_data = pd.read_csv('data/all_variants/myh7_myo5b_dummy_no_nan.csv',
                           sep='\t')

    myo5b_with_genename = all_data.loc[all_data['genename']=='MYO5B']
    myh7_with_genename = all_data.loc[all_data['genename']=='MYH7']

    myo5b_with_genename = myo5b_with_genename.drop(['genename'],axis=1)
    myh7_with_genename = myh7_with_genename.drop(['genename'],axis=1)

    myh7 = read_data_set(myh7_with_genename)
    myo5b = read_data_set(myo5b_with_genename)
    myh7_full = read_data_set(myh7_with_genename,test_size=0)
    myo5b_full = read_data_set(myo5b_with_genename,test_size=0)

    train_fn = myh7_full.train
    train_fn_2 = myo5b.train
    test_fn = myo5b.test

    # constant
    batch_size = 100
    epoch_num = 3000
    lr = 0.0001
    batch_size_2 = 50
    epoch_num_2 = 100
    lr_2 = 0.0001
    batch_per_ep_2 = ceil(train_fn_2.num_examples/batch_size_2)
    batch_per_ep = ceil(train_fn.num_examples/batch_size)


    # ================== feedforward shallow network ===========================
    # model
    inputs = tf.placeholder(tf.float32,(None, train_fn.num_features))
    labels = tf.placeholder(tf.float32,(None, 2))
    fn_outputs,fn_probs = feedforward_net(inputs)
    # loss and training options
    loss_fn = tf.reduce_mean(
                    tf.nn.weighted_cross_entropy_with_logits(
                    targets=labels,
                    logits=fn_outputs,
                    pos_weight=4))
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_fn)
    train_op_2 = tf.train.AdamOptimizer(learning_rate=lr_2).minimize(loss_fn)

    # initializer
    init = tf.global_variables_initializer()

    # start training
    with tf.Session() as sess:
        sess.run(init)
        for ep in range(epoch_num):
            for batch_no in range(batch_per_ep):
                batch_data, batch_label = train_fn.next_batch(batch_size)
                batch_label_onehot = dense_to_one_hot(batch_label,2)
                _, probs,error = sess.run([train_op,
                                           fn_probs,loss_fn],
                                           feed_dict={inputs:batch_data,
                                           labels:batch_label_onehot})
                print('Epoch: {0}\tIteration:{1}\tError: {2}\t'.format(
                ep, batch_no, error
                ))
        for ep in range(epoch_num_2):
            for batch_no in range(batch_per_ep_2):
                batch_data, batch_label = train_fn_2.next_batch(batch_size_2)
                batch_label_onehot = dense_to_one_hot(batch_label,2)
                _, probs,error = sess.run([train_op_2,
                                           fn_probs,loss_fn],
                                           feed_dict={inputs:batch_data,
                                           labels:batch_label_onehot})
                print('Epoch: {0}\tIteration:{1}\tError: {2}\t'.format(
                ep, batch_no, error
                ))

        # # test the trained network
        # batch_data, batch_label = test_fn.next_batch(batch_size=50)
        # batch_label_onehot = dense_to_one_hot(batch_label,2)
        # probs, error = sess.run([fn_probs,
        #                          loss_fn],
        #                          feed_dict={inputs:batch_data,
        #                                     labels:batch_label_onehot})
        #
        # pred = probs[:,1]
        # pred[pred>=0.5] = 1
        # pred[pred<0.5] = 0
        #
        # tn, fp, fn, tp = confusion_matrix(batch_label,pred).ravel()
        # sensitivity = tp/(fn+tp)
        # specificity = tn/(fp+tn)
        # print('Test dataset\tError: {0}'.format(error))
        # print(tn,fp,fn,tp)
        # print(sensitivity,specificity)

        # test
        batch_label_onehot = dense_to_one_hot(test_fn.labels,2)
        _,probs,_ = sess.run([fn_outputs,
                              fn_probs,
                              loss_fn],
                              feed_dict={inputs:test_fn.values,
                                         labels:batch_label_onehot})

        pred = probs[:,1]
        pred[pred>=0.5] = 1
        pred[pred<0.5] = 0

        # tn, fp, fn, tp = confusion_matrix(test_fn.labels,pred).ravel()
        # sensitivity = tp/(fn+tp)
        # specificity = tn/(fp+tn)
        # print(tn,fp,fn,tp)
        # print(sensitivity,specificity)
