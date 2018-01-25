import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
from lib.read_data import dataset,Datasets

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# feature extractors
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB

# finetuning
from sklearn.model_selection import GridSearchCV

# validation
from sklearn import metrics
from sklearn.metrics import confusion_matrix

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
    file_path = os.path.join('data','all_variants','cadd_with_info_no_nan_all_imputed.tsv')
    all_variants = load_cadd_annotation(file_path)
    all_variants['INFO'] = all_variants['INFO'].astype('category')
    all_variants['INFO'] = all_variants['INFO'].cat.codes
    for col in all_variants.select_dtypes(exclude=[np.object]).columns:
        null = all_variants[col].isnull().values.sum()
        if null>0:
            all_variants = all_variants.drop([col],axis=1)
    dummy_all_var=all_variants.drop(['Anc','INFO','chr_pos'],axis=1)
    dummy_all_var = pd.get_dummies(dummy_all_var,sparse=True)
    dummy_all_var['INFO'] = all_variants['INFO']

    # read data
#     processed_all = pd.read_csv('data/myh7/myh7_myo5b.csv',sep='\t')
#     processed_myh7 = processed_all.loc[processed_all['genename']=='MYH7']
#     processed_myo5b = processed_all.loc[processed_all['genename']=='MYO5B']
#     processed_myo5b = processed_myo5b.drop('genename',axis=1) # drop pos
#     processed_myh7 = processed_myh7.drop('genename',axis=1) # drop pos

#     myh7_data = read_data_set(processed_myh7,BENCHMARK=False)
#     myo5b_data = read_data_set(processed_myo5b,BENCHMARK=False)
    full_data = read_data_set(dummy_all_var,BENCHMARK=False)
    print('Dataset loaded.',full_data.train.values.shape)

# # ================model selection==========================================
#     # # PCA + LogisticRegression
#     # # Parameters
#     n_components = np.arange(10,50,10)
#     class_weight = ['balanced']#,{1:2,0:1}]
#     param_grid_logr = [{'pca__n_components':n_components,
#                    'logr__penalty':['l1'],#'l2'],
#                    'logr__C':[2],#3,4,5],
#                    'logr__class_weight':class_weight}]
#     # pipeline
#     pipeline_logr = Pipeline(steps=[('pca',PCA()),
#                                ('logr',LogisticRegression())])
#     # save model
#     filename = os.path.join('model')#,'pca_logr_new.sav')
#     # display results
#     classifier_logr = display_res_gavin_and_best_model(param_grid_logr,
#                                      pipeline_logr,
#                                      myh7_data)#,
#                                      #filename)

#     inference(classifier_logr,myo5b_data)
#     # display gavin results
#     sensitivity_g,specificity_g = read_gavin(test_gavin,myh7_data.test.labels)
#     print('>>> gavin model results: sensitivity: {:.{prec}}\tspecificity: {:.{prec}f}'.\
#     format(sensitivity_g,specificity_g,prec=3))

# ==========================================================
#     # Linear model + SGDregressor
#     # Parameters
#     n_components = np.arange(10,50,10)
#     loss = ['squared_loss', 'huber',
#             'epsilon_insensitive',
#             'squared_epsilon_insensitive']
#     penalty = ['l1','l2','elasticnet']
#     l1_ratio = [0.15,0.2,0.5,0.8]
#     learning_rate = ['constant','optimal','invscaling']
#     param_grid_sgd = [{#'pca__n_components':n_components,
#                    'sgd__penalty':penalty,
#                    'sgd__loss':loss,
#                    'sgd__l1_ratio':l1_ratio,
#                    'sgd__learning_rate':learning_rate,
#                    'sgd__tol':[1e-6],
#                    'sgd__warm_start':[True],
#                    'sgd__max_iter':[10000],
#                    'sgd__eta0':[0.1,0.01,0.5],
#                    'sgd__class_weight':['balanced']}]
#     # pipeline
#     pipeline_sgd = Pipeline(steps=[#('pca',PCA()),
#                                ('sgd',SGDClassifier())])
#     # save model
#     filename = os.path.join('model','pca_logr_new.sav')
#     # display results
#     classifier_sgd = display_res_gavin_and_best_model(param_grid_sgd,
#                                      pipeline_sgd,
#                                      full_data)#,
#                                      #filename)
#     inference(classifier_sgd,full_data)
    # ==========================================================
    # Linear model + SGDregressor
    # Parameters
    n_components = np.arange(10,100,10)
    loss = ['squared_loss', 'huber',
            'epsilon_insensitive',
            'squared_epsilon_insensitive']
    penalty = ['elasticnet']
    l1_ratio = [0.2]
    learning_rate = ['optimal','invscaling']
    param_grid_sgd = [{#'pca__n_components':n_components,
                   'sgd__penalty':penalty,
                   'sgd__loss':loss,
                   'sgd__l1_ratio':l1_ratio,
                   'sgd__learning_rate':learning_rate,
                   'sgd__tol':[1e-6],
                   'sgd__warm_start':[True],
                   'sgd__max_iter':[10000],
                   'sgd__eta0':[0.1],#0.01,0.5],
                   'sgd__class_weight':['balanced']}]
    # pipeline
    pipeline_sgd = Pipeline(steps=[#('pca',PCA()),
                               ('sgd',SGDClassifier())])
    # save model
    filename = os.path.join('model','pca_logr_new.sav')
    # display results
    classifier_sgd = display_res_gavin_and_best_model(param_grid_sgd,
                                     pipeline_sgd,
                                     full_data)#,
                                     #filename)
    inference(classifier_sgd,full_data)