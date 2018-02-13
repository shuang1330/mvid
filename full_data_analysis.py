
from math import ceil
# from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import os
import numpy as np
from lib.read_data import dataset,Datasets

import pandas as pd
import numpy as np
import os

# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
#
# # feature extractors
# from sklearn.decomposition import PCA
# from sklearn.ensemble import RandomForestClassifier
#
# # classifiers
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import ElasticNet
# from sklearn.svm import SVC
# from sklearn.linear_model import SGDClassifier
# from sklearn.linear_model import PassiveAggressiveClassifier
# from sklearn.linear_model import Perceptron
# from sklearn.naive_bayes import MultinomialNB

# finetuning
# from sklearn.model_selection import GridSearchCV

# validation
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix

import tensorflow.contrib.layers as lays
import tensorflow as tf
from lib.net import autoencoder,feedforward_net
# from sklearn.metrics import confusion_matrix

def dense_to_one_hot(labels_dense, num_classes):
  """
  Convert class labels from scalars to one-hot vectors.
  """
  return np.eye(num_classes)[np.array(labels_dense).reshape(-1)]


def load_cadd_annotation(file_path):
    '''
    load original cadd annotation
    '''
    dtype = {'#Chrom':np.object,
            'isDerived':np.object,
            'motifEName':np.object,
            'motifEHIPos':np.object,
            'PolyPhenCat':np.object,
            'SIFTcat':np.object}
    cadd_anno = pd.read_csv(file_path,sep='\t',dtype=dtype)
    print('Cadd annotation loaded.')
    return cadd_anno

def read_data_set(data_table,test_size=0.25,BENCHMARK=False):
    '''
    convert a pandas dataframe data table into Datasets(dataset,dataset)
    '''
    train=data_table.sample(frac=(1-test_size),random_state=200)
    test=data_table.drop(train.index)
    # train, test = train_test_split(data_table,test_size=0.25)
    train_x = train[[col for col in train.columns
    if col not in ['INFO','gavin_res']]]
    features = train_x.columns
    train_x = np.array(train_x)
    test_x = np.array(test[[col for col in train.columns
    if col not in ['INFO','gavin_res']]])
    train_y = np.array(train['INFO'],dtype=np.int8)
    test_y = np.array(test['INFO'],dtype=np.int8)

    if BENCHMARK:
        return Datasets(train=dataset(train_x,train_y,features),
                        test=dataset(test_x,test_y,features)),\
                        train['gavin_res'],\
                        test['gavin_res']
    return Datasets(train=dataset(train_x,train_y,features),
                    test=dataset(test_x,test_y,features))

def run_display_output(classifier,test,DRAW=False):
    '''
    get confusion matrix and auc score for test dataset
    (optional) draw roc curve
    '''
    pred = classifier.predict(test.values)
    pred[pred>0.5] = 1
    pred[pred<0.5] = 0
    tn, fp, fn, tp = confusion_matrix(test.labels,pred).ravel()#confusion matrix
    print(tn,fp,fn,tp)
    sensitivity = tp/(fn+tp)
    specificity = tn/(fp+tn)
    prods = classifier.predict(test.values)
    if hasattr(classifier,'predict_proba'):
        prods = classifier.predict_proba(test.values)[:,1]
    else:
        probs = classifier.predict(test.values)
    fpr, tpr, _ = metrics.roc_curve(test.labels, prods)
    score = metrics.auc(fpr,tpr) #auc score
    if DRAW:
        draw_roc_curve(fpr,tpr,score)

    return sensitivity, specificity, score

def display_res_gavin_and_best_model(param_grid,pipeline,mvid,filename=None):
    '''
    use model defined by pipeline to fit mvid Dataset
    gridsearchCV determine the parameters given in param_grid
    (optional) save the model in path given in filename
    '''
    classifier = GridSearchCV(estimator=pipeline,
                              param_grid=param_grid)

    print('Start training...')
    classifier.fit(mvid.train.values,mvid.train.labels)
    print('Model Description:\n',classifier.best_estimator_)
    if filename:
        pickle.dump(classifier,open(filename,'wb'))
        print('Saved model to path:',filename)
    sensitivity,specificity,score = run_display_output(classifier,mvid.test)
    print('>>> best model results: sensitivity: {:.{prec}}\tspecificity: {:.{prec}f}\tauc:{}'.\
    format(sensitivity,specificity,score,prec=3))
    return classifier

def incremental_training(param_grid,pipeline,data,filename=None):
    classifier = GridSearchCV(estimator=pipeline,
                              param_grid=param_grid)
    print('Start training...')
    classifier.fit()

def inference(classifier,dataset):
    sensitivity,specificity,score = run_display_output(classifier,dataset.test)
    print('>>> transfer to myo5b gene: sensitivity: {:.{prec}}\tspecificity: {:.{prec}f}\tauc:{}'.\
    format(sensitivity,specificity,score,prec=3))

def read_gavin(gavin_res, labels):
    '''
    compare gavin results with labels for a certain subset of data
    '''
    gavin_res = gavin_res.replace('Pathogenic',1)
    gavin_res = gavin_res.replace('Benign',0)
    tn_g, fp_g, fn_g, tp_g = \
    confusion_matrix(labels, gavin_res.astype(np.int8)).ravel()
    sensitivity_g = tp_g/(fn_g+tp_g)
    specificity_g = tn_g/(fp_g+tn_g)
    return sensitivity_g, specificity_g

if __name__ == '__main__':
    file_path = os.path.join('data','all_variants',
                             'cadd_with_info_no_nan_all_imputed.tsv')
    all_variants = load_cadd_annotation(file_path)
    # print(all_variants.columns)
    # print(all_variants.head())
    # raise NotImplementedError

    all_variants['INFO'] = all_variants['INFO'].astype('category')
    all_variants['INFO'] = all_variants['INFO'].cat.codes
    for col in all_variants.select_dtypes(exclude=[np.object]).columns:
        null = all_variants[col].isnull().values.sum()
        if null>0:
            all_variants = all_variants.drop([col],axis=1)
    dummy_all_var = all_variants.drop(['Anc','INFO','chr_pos','key','POS'],axis=1)
    dummy_all_var = pd.get_dummies(dummy_all_var,sparse=True)
    dummy_all_var['INFO'] = all_variants['INFO']
    # print(dummy_all_var.columns)

    # read data
    # full_data = read_data(dummy_all_var,BENCHMARK=False)
    mvid = read_data_set(dummy_all_var,BENCHMARK=False)
    print('Dataset loaded.',mvid.train.values.shape)

    # constant
    batch_size = 1000
    epoch_num = 11
    lr = 0.001
    batch_per_ep = ceil(mvid.train.num_examples/batch_size)

    train_fn = mvid.train
    test_fn = mvid.test


    # ======================= feedfoward networks ==============================
    # model
    inputs = tf.placeholder(tf.float32,(None, train_fn.num_features))
    labels = tf.placeholder(tf.float32,(None, 2))
    # ae_outputs = autoencoder(inputs)
    fn_outputs,fn_probs = feedforward_net(inputs)
    # loss and training options
    loss_fn = tf.reduce_mean(
                    tf.nn.weighted_cross_entropy_with_logits(
                    targets=labels,
                    logits=fn_outputs,
                    pos_weight=4))
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_fn)

    # initializer
    init = tf.global_variables_initializer()

    # start training
    with tf.Session() as sess:
        sess.run(init)
        for ep in range(epoch_num):
            for batch_no in range(batch_per_ep):
                batch_data, batch_label = train_fn.next_batch(batch_size)
                batch_label_onehot = dense_to_one_hot(batch_label,2)
                _, recon_data,error = sess.run([train_op,
                                                fn_outputs,loss_fn],
                                                feed_dict={inputs:batch_data,
                                                labels:batch_label_onehot})
                print('Epoch: {0}\tIteration:{1}\tError: {2}\t'.format(
                ep, batch_no, error
                ))
                print('Epoch: {0}\tIteration:{1}\tError: {2}\t'.format(
                ep, batch_no, error
                ))

        # test the trained network
        batch_data, batch_label = test_fn.values, test_fn.labels
        # print(batch_data.shape, batch_label.shape)
        batch_label_onehot = dense_to_one_hot(batch_label,2)
        recon_data,error, probs = sess.run([fn_outputs,
                                            loss_fn,fn_probs],
                                            feed_dict={inputs:batch_data,
                                            labels:batch_label_onehot})

        # print(batch_label_onehot[:,0])
        probs = probs[:,0]
        probs[probs>0.5] = 1
        probs[probs<0.5] = 0
        # print(probs)
        tn, fp, fn, tp = confusion_matrix(batch_label_onehot[:,0],
                                          probs).ravel()
        print('Test dataset\tError: {0}'.format(error))
        print(tn, fp, fn, tp)



    # # pca LogisticRegression
    # pipeline_pca = Pipeline(
    # steps=[('pca',PCA()),
    # ('logr',LogisticRegression())]
    # )
    # param_grid_pca = [
    # {
    # 'pca__n_components':np.arange(50,100,50),
    # 'logr__class_weight':['balanced']
    # }
    # ]
    # classifier_sgd = display_res_gavin_and_best_model(param_grid_pca,
    #                                  pipeline_pca,
    #                                  full_data)#,
    #                                  #filename)
    # inference(classifier_sgd,myo5b.test)

    # ==========================================================
    # Linear model + SGDregressor
    # Parameters
#     n_components = np.arange(10,100,10)
#     loss = ['squared_loss']#, 'huber',
# #            'epsilon_insensitive',
# #            'squared_epsilon_insensitive']
#     penalty = ['elasticnet']
#     l1_ratio = [0.2]
#     learning_rate = ['optimal','invscaling']
    # pipeline
    # param_grid_sgd = [{#'pca__n_components':n_components,
    #                'sgd__penalty':penalty,
    #                'sgd__loss':loss,
    #                'sgd__l1_ratio':l1_ratio,
    #                'sgd__learning_rate':learning_rate,
    #                'sgd__tol':[1e-6],
    #                'sgd__warm_start':[True],
    #                'sgd__max_iter':[10000],
    #                'sgd__eta0':[0.1],#0.01,0.5],
    #                'sgd__class_weight':['balanced']}]
    # # pipeline
    # pipeline_sgd = Pipeline(steps=[#('pca',PCA()),
    #                            ('sgd',SGDClassifier())])

    # # incremental training
    # all_classes = np.array([0,1])
    # class_weight = compute_class_weight('balanced',
    #                                     all_classes,
    #                                     full_data.test.labels)
    # weight_dic = {all_classes[0]:class_weight[0],
    #               all_classes[1]:class_weight[1]*2}
    # classifier_sgd = SGDClassifier(loss='squared_epsilon_insensitive',
    #                                penalty='elasticnet',
    #                                l1_ratio=0.2,
    #                                learning_rate='optimal',
    #                                warm_start=True,
    #                                eta0=0.01,
    #                                tol=1e-06,
    #                                class_weight=weight_dic)
    # epoch_num = 11
    # batch_size=10000
    # batch_per_ep = ceil(full_data.train.num_examples/batch_size)
    # print(batch_per_ep)
    #
    # full_data.train._epochs_completed = 0
    # for ep in range(epoch_num):
    #     full_data.train._epochs_completed += 1
    #     for batch_no in range(batch_per_ep):
    #         full_data.train._index_in_epoch = batch_no
    #         print('Epoch: {0}\tIteration:{1}\t'.format(ep,batch_no))
    #         batch_values,batch_labels = \
    #         full_data.train.next_batch(batch_size)
    #         classifier_sgd.partial_fit(batch_values,
    #                                    batch_labels,
    #                                    classes=all_classes)
    #         pred = classifier_sgd.predict(full_data.test.values)
    #         tn, fp, fn, tp = confusion_matrix(full_data.test.labels,pred).ravel()
    #         print('TN: {0} FP: {1} Fn: {2} TP: {3}'.format(tn, fp, fn, tp))
    #         sensitivity,specificity,score = run_display_output(classifier_sgd,full_data.test)
    #         print('>>> {:.{prec}}\tspecificity: {:.{prec}f}\tauc:{}'.\
    #         format(sensitivity,specificity,score,prec=3))
    #
    #
    #
    # # save model
    # filename = os.path.join('model','incre_linear_fulldata.sav')
    # # display results
