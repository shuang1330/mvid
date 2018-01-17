'''
take the original pandas table
preprocess the table
save the processed table in csv format
'''

import pandas as pd
import os
from os.path import isfile, join
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


if __name__=='__main__':
    # read data file
    datafile = os.path.join('data','mhy7.tsv')
    datatable_pd = pd.read_csv(datafile,sep='\t')
    datatable_pos = datatable_pd['POS']

    # delete some columns that were not used in cadd paper
    del_cols = ['CHROM','POS','ID','isDerived','AnnoType','ConsScore',
                'ConsDetail','mapAbility20bp','mapAbility35bp',
                'scoreSegDup','isKnownVariant','ESP_AF','ESP_AFR',
                'ESP_EUR','TG_AF','TG_ASN','TG_AMR','TG_AFR','TG_EUR',
                'GeneID','FeatureID','CCDS','GeneName','Exon',
                'Intron','RawScore']
    datatable_pd = datatable_pd.drop(columns=del_cols)

    # delete columns without a single value
    datatable_pd = datatable_pd.dropna(axis=1,how='all')

    # fill in values recommended by cadd paper
    values = {'GerpRS':0, 'GerpRSpval':1,'EncExp':0,'EncOCC':5,
              'EncOCCombPVal':0,'EncOCDNasePVal':0,'EncOCFairePVal':0,
              'EncOCpolIIPVal':0,'EncOCctcfPVal':0,'EncOCmycPVal':0,
              'EncOCDNaseSig':0,'EncOCFaireSig':0,'EncOCpolIISig':0,
              'EncOCctcfSig':0,'EncOCmycSig':0,'tOverlapMotifs':0,
              'motifDist':0,'TFBS':0,'TFBSPeaksMax':0,'PolyPhenVal':0,
              'SIFTval':0,'TFBSPeaks':0}
    datatable_pd = datatable_pd.fillna(values)

    # transform objects to dummies
    categorical_feature_names = \
    datatable_pd.select_dtypes(include=np.object).columns
    categories={} # contains all the levels in those feature columns
    for f in categorical_feature_names:
        datatable_pd[f] = datatable_pd[f].astype('category')
        categories[f] = datatable_pd[f].cat.categories

    dummy_data = pd.get_dummies(datatable_pd,columns=[col for col in
                                                      categorical_feature_names
                                                      if col not in ['INFO']])
    # change info column into scalar column
    dummy_data['INFO'] = datatable_pd['INFO'].astype('category').cat.codes

    # drop nan values -TODO
    dummy_data_del_all_nan = dummy_data.copy()
    for col in dummy_data.columns:
        null = dummy_data[col].isnull().values.ravel().sum()
        if null > 0:
            dummy_data_del_all_nan = dummy_data_del_all_nan.drop(columns=col)

    # normalized the numerical values before any processing afterwards
    min_max_scaler = MinMaxScaler()
    dummy_data_scaled = min_max_scaler.fit_transform(dummy_data_del_all_nan)
    dummy_data_scaled = pd.DataFrame(dummy_data_scaled,
                                     columns=dummy_data_del_all_nan.columns)

    # save the preprocessed data as csv file
    dummy_data_scaled['POS'] = datatable_pos
    dummy_data_scaled.to_csv('data/dummy_no_nan_data.csv',sep='\t',index=False)
