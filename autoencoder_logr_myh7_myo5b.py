'''
autoencoder and conv classifiers
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

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

def dense_to_one_hot(labels_dense, num_classes):
  """
  Convert class labels from scalars to one-hot vectors.
  """
  return np.eye(num_classes)[np.array(labels_dense).reshape(-1)]
#
# def read_data_set(data_table,test_size=0.25):
#     '''
#     convert a pandas dataframe data table into Datasets(dataset,dataset)
#     '''
#     train, test = train_test_split(data_table,test_size=0.25)
#     train_x = np.array(train[[col for col in train.columns
#     if col not in ['INFO','gavin_res']]])
#     test_x = np.array(test[[col for col in train.columns
#     if col not in ['INFO','gavin_res']]])
#     train_y = np.array(train['INFO'],dtype=np.int8)
#     test_y = np.array(test['INFO'],dtype=np.int8)
#     return Datasets(train=dataset(train_x,train_y),
#     test=dataset(test_x,test_y))

def read_data_set(data_table,test_size=0.25,BENCHMARK=False):
    '''
    convert a pandas dataframe data table into Datasets(dataset,dataset)
    '''
    train, test = train_test_split(data_table,test_size=0.25)
    train_x = train[[col for col in train.columns
    if col not in ['INFO','gavin_res']]]
    train_x = np.array(train_x)
    test_x = np.array(test[[col for col in train.columns
    if col not in ['INFO','gavin_res']]])
    train_y = np.array(train['INFO'],dtype=np.int8)
    test_y = np.array(test['INFO'],dtype=np.int8)

    if BENCHMARK:
        return Datasets(train=dataset(train_x,train_y),
                        test=dataset(test_x,test_y)),\
                        train['gavin_res'],\
                        test['gavin_res']
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

    train_ae = myh7_full.train
    test_ae = myh7_full.train
    train_logr = myh7.train
    test_logr = myh7.test
    # train_ae_2 = myo5b.train


    # constant
    batch_size = 50
    epoch_num = 200
    lr = 0.001
    batch_size_2 = 50
    epoch_num_2 = 50
    lr_2 = 0.0005
    batch_per_ep = ceil(train_ae.num_examples/batch_size)
    # batch_per_ep2 = ceil(train_ae_2.num_examples/batch_size_2)



    # ======================== autoencoder ==============================
    # model
    inputs = tf.placeholder(tf.float32,(None, train_ae.num_features))
    labels = tf.placeholder(tf.float32,(None, 2))
    ae_outputs,ae_bottle = autoencoder(inputs)
    # loss and training options
    loss_ae = tf.reduce_mean((tf.square(ae_outputs-inputs))) # autoencoder
    # loss_fn = tf.reduce_mean((tf.square(fn_outputs-labels)))
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_ae)
    train_op_2 = tf.train.AdamOptimizer(learning_rate=lr_2).minimize(loss_ae)

    # initializer
    init = tf.global_variables_initializer()

    # start training
    with tf.Session() as sess:
        sess.run(init)
        for ep in range(epoch_num):
            for batch_no in range(batch_per_ep):
                batch_data, batch_label = train_ae.next_batch(batch_size)
                batch_label_onehot = dense_to_one_hot(batch_label,2)
                _, recon_data,error = sess.run([train_op,
                                                ae_outputs,loss_ae],
                                                feed_dict={inputs:batch_data,
                                                labels:batch_label_onehot})
                print('Epoch: {0}\tIteration:{1}\tError: {2}\t'.format(
                ep, batch_no, error
                ))
        # for ep in range(epoch_num_2):
        #     for batch_no in range(batch_per_ep2):
        #         batch_data, batch_label = train_ae.next_batch(batch_size)
        #         batch_label_onehot = dense_to_one_hot(batch_label,2)
        #         _, recon_data,error = sess.run([train_op_2,
        #                                         ae_outputs,loss_ae],
        #                                         feed_dict={inputs:batch_data,
        #                                         labels:batch_label_onehot})
        #         print('Epoch: {0}\tIteration:{1}\tError: {2}\t'.format(
        #         ep, batch_no, error
        #         ))

        # test the trained network
        batch_data, batch_label = test_ae.next_batch(batch_size=50)
        batch_label_onehot = dense_to_one_hot(batch_label,2)
        recon_data, bottle, error = sess.run([ae_outputs,
                                              ae_bottle,
                                              loss_ae],
                                              feed_dict={inputs:batch_data,
                                              labels:batch_label_onehot})
        print('Test dataset\tError: {0}'.format(error))

        # apply LogisticRegression
        batch_label_onehot = dense_to_one_hot(train_logr.labels,2)
        _,train_logr_features,_ = sess.run([ae_outputs,
                                          ae_bottle,
                                          loss_ae],
                                          feed_dict={inputs:train_logr.values,
                                          labels:batch_label_onehot})
        batch_label_onehot = dense_to_one_hot(test_logr.labels,2)
        _,test_logr_features,_ = sess.run([ae_outputs,
                                         ae_bottle,
                                         loss_ae],
                                         feed_dict={inputs:test_logr.values,
                                         labels:batch_label_onehot})

        # GridSearchCV + Parameters
        class_weight = ['balanced']
        param_grid_logr = [{'logr__penalty':['l1','l2'],
                            'logr__C':[1,2,3,4,5],
                            'logr__class_weight':class_weight}]
        # pipeline
        pipeline_logr = Pipeline(steps=[('logr',LogisticRegression())])
        # display results
        classifier_logr = GridSearchCV(estimator=pipeline_logr,
                                       param_grid=param_grid_logr)

        print('Start training...')
        classifier_logr.fit(train_logr_features,train_logr.labels)
        print('Model Description:\n',classifier_logr.best_estimator_)
        pred = classifier_logr.predict(test_logr_features)
        tn, fp, fn, tp = confusion_matrix(test_logr.labels,pred).ravel()
        sensitivity = tp/(fn+tp)
        specificity = tn/(fp+tn)
        print('>>> best model results: sensitivity: {:.{prec}}\tspecificity: {:.{prec}f}'.\
        format(sensitivity,specificity,prec=3))



        # # test on myo5b dataset
        # batch_label_onehot = dense_to_one_hot(myo5b.train.labels,2)
        # _,train_ae_features,_ = sess.run([ae_outputs,ae_bottle, loss_ae],
        # feed_dict={inputs:myo5b.train.values,labels:batch_label_onehot})
        # pred = classifier_logr.predict(train_ae_features)
        # tn, fp, fn, tp = confusion_matrix(myo5b.train.labels,pred).ravel()
        # sensitivity = tp/(fn+tp)
        # specificity = tn/(fp+tn)
        # print(tn,fp,fn,tp)
        # print(sensitivity,specificity)
