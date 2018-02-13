import numpy as np
import pandas as pd
import pickle
import os
from lib.read_data import dataset,Datasets

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# feature extractors
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
# finetuning
from sklearn.model_selection import GridSearchCV
# validation
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# import matplotlib.pyplot as plt
# TODO: support matplotlib drawing

def read_data_set(data_table,test_size=0.25,BENCHMARK=False):
    '''
    convert a pandas dataframe data table into Datasets(dataset,dataset)
    '''
    train, test = train_test_split(data_table,test_size=0.25)
    train_x = train[[col for col in train.columns
    if col not in ['INFO','gavin_res']]]
    features = train_x.columns
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

def draw_roc_curve(fpr,tpr,score):
    '''
    draw roc curve
    '''
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % score)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()

def run_display_output(classifier,test,DRAW=False):
    '''
    get confusion matrix and auc score for test dataset
    (optional) draw roc curve
    '''
    pred = classifier.predict(test.values)
    tn, fp, fn, tp = confusion_matrix(test.labels,pred).ravel()#confusion matrix
    print(tn,fp,fn,tp)
    sensitivity = tp/(fn+tp)
    specificity = tn/(fp+tn)
    prods = classifier.predict_proba(test.values)[:,1]
    fpr, tpr, _ = metrics.roc_curve(test.labels, prods)
    score = metrics.auc(fpr,tpr) #auc score
    if DRAW:
        draw_roc_curve(fpr,tpr,score)

    return sensitivity, specificity, score

def run_display_output_elasticnet(classifier,test,DRAW=False):
    '''
    get confusion matrix and auc score for test dataset
    (optional) draw roc curve
    '''
    pred = classifier.predict(test.values)
    pass
    # tn, fp, fn, tp = confusion_matrix(test.labels,pred).ravel()#confusion matrix
    # print(tn,fp,fn,tp)
    # sensitivity = tp/(fn+tp)
    # specificity = tn/(fp+tn)
    # prods = classifier.predict_proba(test.values)[:,1]
    # fpr, tpr, _ = metrics.roc_curve(test.labels, prods)
    # score = metrics.auc(fpr,tpr) #auc score
    # if DRAW:
    #     draw_roc_curve(fpr,tpr,score)
    #
    # return sensitivity, specificity, score

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

def display_res_gavin_and_elasticnet(param_grid,pipeline,mvid,filename=None):
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
    sensitivity,specificity,score = run_display_output_elasticnet(classifier,mvid.test)
    print('>>> best model results: sensitivity: {:.{prec}}\tspecificity: {:.{prec}f}\tauc:{}'.\
    format(sensitivity,specificity,score,prec=3))
    return classifier


if __name__=='__main__':

    # read data
    all_data = pd.read_csv('data/all_variants/myh7_myo5b_dummy_no_nan_with_gavin.csv',
                           sep='\t')
    myo5b_with_genename = all_data.loc[all_data['genename']=='MYO5B']
    myh7_with_genename = all_data.loc[all_data['genename']=='MYH7']

    myo5b_with_genename = myo5b_with_genename.drop(['genename'],axis=1)
    myh7_with_genename = myh7_with_genename.drop(['genename'],axis=1)

    myh7, _, myh7_gavin = read_data_set(myh7_with_genename,BENCHMARK=True)
    myo5b, myo5b_gavin,_ = read_data_set(myo5b_with_genename,test_size=0,BENCHMARK=True)

    # mvid,train_gavin, test_gavin = read_data_set(data,BENCHMARK=True)
    # print(data.head())
    # raise NotImplementedError # check the dataset loaded
    print('Dataset loaded.',myo5b.train.labels.shape)
    print(myo5b_gavin.shape)

# ================model selection==========================================
    # # # PCA + LogisticRegression
    # # # Parameters
    # n_components = np.arange(2,myo5b.train.num_features,10)
    # class_weight = ['balanced',{1:4,0:1},{1:2,0:1}]
    # param_grid_logr = [{'pca__n_components':n_components,
    #                'logr__penalty':['l1','l2'],
    #                'logr__C':[1,2,3,4,5],
    #                'logr__class_weight':class_weight}]
    # # pipeline
    # pipeline_logr = Pipeline(steps=[('pca',PCA()),
    #                            ('logr',LogisticRegression())])
    # # save model
    # filename = os.path.join('model','pca_logr.sav')
    # # display results
    # classifier_logr = display_res_gavin_and_best_model(param_grid_logr,
    #                                  pipeline_logr,
    #                                  myo5b)

    # # PCA + ElasticNet
    # pipeline_eln = Pipeline(steps=[('pca',PCA()),
    #                                ('eln',ElasticNet())])
    # alpha = [0.2,0.4,0.6,1]
    # l1_ratio = [0.2,0.4,0.6,0.8]
    # normalize=[True]
    # param_grid_eln = [{'pca__n_components':n_components,
    #                    'eln__alpha':alpha,
    #                    'eln__l1_ratio':l1_ratio,
    #                    'eln__normalize':normalize}]
    # display_res_gavin_and_elasticnet(param_grid_eln,
    #                                  pipeline_eln,
    #                                  mvid)

    # # PCA + RandomForest
    # pipeline_ranfor = Pipeline(steps=[('pca',PCA()),
    #                                   ('ranfor',RandomForestClassifier())])
    # n_estimators = [10,50,100]
    # param_grid_ranfor = [{'pca__n_components':n_components,
    #                       'ranfor__n_estimators':n_estimators,
    #                       'ranfor__class_weight':class_weight}]
    # filename = os.path.join('model','pca_ranfor.sav')
    # classifier_ranfor = display_res_gavin_and_best_model(param_grid_ranfor,
    #                                  pipeline_ranfor,
    #                                  myo5b)

    # display gavin results
    sensitivity_g,specificity_g = read_gavin(myo5b_gavin,myo5b.train.labels)
    print('>>> gavin model results: sensitivity: {:.{prec}}\tspecificity: {:.{prec}f}'.\
    format(sensitivity_g,specificity_g,prec=3))
# ======================================================================
