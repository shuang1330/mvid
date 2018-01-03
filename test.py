#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:54:48 2017
@author: paulhuizinga
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

#%%

folder = "data/"
file_name = folder + "myo5b_variants_patho_benign_cadd1.3fullannot_v1.xlsx"

#df = pd.read_excel(file)
datafile= pd.ExcelFile(file_name)
df = datafile.parse(datafile.sheet_names[0])

# create class column
df.loc[:,'CLASS'] = df['INFO']

# move class column to first position
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

# remove column info
df = df.drop('INFO', 1)

# # make pie chart of class distribution
# plt.pie(df['CLASS'].value_counts(), shadow=False, labels=set(df.iloc[:,0]), autopct='%1.1f%%')
# plt.title('Class distribution')
# plt.show()

#%%

# Write basic statistics to list
def BasicStatistics(dataframe):
    statframe = []
    statlist = []
    cols = dataframe.columns.tolist()
    for i in range(len(dataframe.columns)):
        if dataframe.iloc[:,i].dtypes == 'float' or dataframe.iloc[:,i].dtypes == 'int':
            statlist.append([cols[i], 'numeric', \
                             dataframe.iloc[:,i].min(), \
                             dataframe.iloc[:,i].max(), \
                             dataframe.iloc[:,i].max() - dataframe.iloc[:,i].min(), \
                             dataframe.iloc[:,i].mean(), \
                             dataframe.iloc[:,i].std(), \
                             dataframe.iloc[:,i].count(), \
                             dataframe.iloc[:,i].sum(), \
                             dataframe.iloc[:,i].nunique(), \
                             dataframe.iloc[:,i].isnull().sum()])
        elif dataframe.iloc[:,i].dtypes == 'object':
            statlist.append([cols[i],'string', \
                             '' ,'' ,'' ,'' ,'' , \
                             dataframe.iloc[:,i].count(),'' , \
                             dataframe.iloc[:,i].nunique(), \
                             set(dataframe.iloc[:,i]), \
                             dataframe.iloc[:,i].isnull().sum()])
        elif dataframe.iloc[:,i].dtypes == 'bool':
            statlist.append([cols[i],'boolean', \
                             '' ,'' ,'' ,'' ,'' , \
                             dataframe.iloc[:,i].count(), \
                             '' , \
                             dataframe.iloc[:,i].nunique(), \
                             dataframe.iloc[:,i].isnull().sum()])
    columns = (['column', \
                'type', \
                'min', \
                'max', \
                'range', \
                'mean', \
                'stdev', \
                'aantal', \
                'sum', \
                'unique', \
                'uniquelist', \
                'nan'])

    # Make dataframe
    statframe = pd.DataFrame(statlist, columns=columns)
    return statframe

sf = BasicStatistics(df)

#%%

# remove unpopulated attributes en attributes with only one unique value
columnlist=sf.loc[(sf["unique"] > 1) & (sf["aantal"] > 0), "column"].tolist()

# create list with column names to be removed.
customremovelist = ['PHRED','RawScore']
# result excluded of features phred and rawscore is better then included! why?

# remove columns
for x in customremovelist:
    if x in columnlist: columnlist.remove(x)

# make dataframe with filtered columns
df2 = df[columnlist].reset_index()

# also remove records from statlist
for x in customremovelist:
    #statframe[statframe.column != x]
    tmp = sf[sf["column"] != x]

#%%

#thresholds: (this can be more advanced.....)
#if nan% > 30%: exclude
max_nan = .3

totalrecords = df2.shape[0]

for x in df2.columns.tolist():
    #tel nan waarden in betreffende kolom
    if (df2[x].isnull().sum()/totalrecords) > max_nan:
        # remove column from dataframe
        del df2[x]
    elif df2[x].dtypes == 'float' or df2[x].dtypes == 'int':
        # replace NaN with 0: NOT CORRECT, ADJUST THIS LATER!!
        df2[x]=df2[x].fillna(0)


# Split features and classes
features = df2.iloc[:,2:]
classes = df2.iloc[:,1]


# label encoding for all categorical features
encoder = LabelEncoder()
for x in features.columns.tolist():
    if features[x].dtype == 'object':
        #print(x)
        features[x]=features[x].fillna('not applicable')
        features[x]=encoder.fit_transform(features[x].astype('str'))

# Make test and train sets
# because of the unbalanced dataset use stratify on classes
X_train, X_test, y_train, y_test = train_test_split(features, classes, stratify=classes, random_state=0)#, train_size=.5)

'''
from sklearn.preprocessing import StandardScaler
scale = StandardScaler(with_mean=True, with_std=True, copy=False)
scale.fit(X_train)
scale.transform(X_train)
scale.transform(X_test)
'''

# Parameters
C_range = [1, 2, 3, 4, 5]
penalty_range = ['l1', 'l2']
tolerance_range = [0.000001, 0.00001, 0.0001]
cw_range = ['balanced']
pca_component_range = [20,60]

param_grid = [{'lr__C': C_range
               ,'lr__penalty': penalty_range
               ,'lr__tol': tolerance_range
               ,'lr__class_weight': cw_range
               ,'pca__n_components': pca_component_range
               }]

# Make pipeline (logistic regression)
pipe_lr = Pipeline(steps=[
    #('norm', Normalizer(norm='l1')),
    ('pca', PCA()),
    ('lr', LogisticRegression())
])

classifier = GridSearchCV(estimator=pipe_lr,
                    param_grid=param_grid)

classifier.fit(X_train,y_train)

print('best parameters')
print(classifier.best_estimator_)


def run_output(featureset, classset, algo, dataset):
    predictions = classifier.predict(featureset)
    print(algo + ': ' + dataset)
    print(classification_report(classset, predictions))
    print('confusion matrix')
    print(confusion_matrix(classset, predictions))

run_output(X_train, y_train, 'logistic regression', 'trainset')

run_output(X_test, y_test, 'logistic regression', 'testset')

run_output(features, classes, 'logistic regression', 'total dataset')

#%%

# add classifier results to original dataset
# =============================================================================
# predictions = classifier.predict(features)
# y_test['classifier_result'] = pd.DataFrame(predictions, columns={'classifier_result'})
# 
# df_out = pd.merge(df,y_test['classifier_result'],how = 'left',left_index = True, right_index = True)
# 
# =============================================================================

# uitzoeken welke features of principal components de meest voorspellende waarde hebben.
# resultaat testen op 'grote' dataset.
#%%

probs = classifier.predict_proba(X_test)
y_test[y_test=='Benign'] = 0
y_test[y_test=='Pathogenic'] = 1
fpr, tpr, _ = roc_curve(list(y_test),probs[:,1])
score = auc(fpr,tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
#%%