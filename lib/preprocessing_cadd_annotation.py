import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .read_data import check_null

def drop_impute_full(datatable_pd):
    '''
    drop columns not used by cadd paper
    impute accoding to cadd paper
    drop columns that I do not know how to impute
    '''
    print('The original data table has shape:',datatable_pd.shape)

    if hasattr(datatable_pd,'chr_pos'):
        datatable_pd = datatable_pd.drop(['chr_pos'],axis=1)
    # delete some columns that were not used in cadd paper
    del_cols = ['#Chrom','Anc','Pos','isDerived','AnnoType','ConsScore',
                'ConsDetail','mapAbility20bp','mapAbility35bp',
                'scoreSegDup','isKnownVariant','ESP_AF','ESP_AFR',
                'ESP_EUR','TG_AF','TG_ASN','TG_AMR','TG_AFR','TG_EUR',
                'GeneID','FeatureID','CCDS','GeneName','Exon',
                'Intron']
    datatable_pd = datatable_pd.drop(columns=del_cols)
    print('Deleted features that were not used by cadd paper.')

    # delete columns without a single value
    datatable_pd = datatable_pd.dropna(axis=1,how='all')
    print('Deleted features without a single row value.')

    # fill in values recommended by cadd paper
    values = {'GerpRS':0, 'GerpRSpval':1,'EncExp':0,'EncOCC':5,
              'EncOCCombPVal':0,'EncOCDNasePVal':0,'EncOCFairePVal':0,
              'EncOCpolIIPVal':0,'EncOCctcfPVal':0,'EncOCmycPVal':0,
              'EncOCDNaseSig':0,'EncOCFaireSig':0,'EncOCpolIISig':0,
              'EncOCctcfSig':0,'EncOCmycSig':0,'tOverlapMotifs':0,
              'motifDist':0,'TFBS':0,'TFBSPeaksMax':0,'PolyPhenVal':0,
              'SIFTval':0,'TFBSPeaks':0}
    datatable_pd = datatable_pd.fillna(values)
    print('Impute with recommended values in cadd paper.')

    # drop nan values -TODO
    print('Deleted columns that I do not know how to impute:')
    for col in datatable_pd.select_dtypes(exclude=[np.object]).columns:
        null = datatable_pd[col].isnull().values.ravel().sum()
        if null > 0:
            print(null,col)
            datatable_pd = datatable_pd.drop(columns=col)

    return datatable_pd

def drop_impute_myo5b(datatable_pd):
    '''
    drop columns not used by cadd paper
    impute accoding to cadd paper
    drop columns that I do not know how to impute
    '''
    if hasattr(datatable_pd,'chr_pos'):
        datatable_pd = datatable_pd.drop(['chr_pos'],axis=1)
    # delete some columns that were not used in cadd paper
    del_cols = ['CHROM','POS','ID','isDerived','AnnoType','ConsScore',
                'ConsDetail','mapAbility20bp','mapAbility35bp',
                'scoreSegDup','isKnownVariant','ESP_AF','ESP_AFR',
                'ESP_EUR','TG_AF','TG_ASN','TG_AMR','TG_AFR','TG_EUR',
                'GeneID','FeatureID','CCDS','GeneName','Exon',
                'Intron']
    datatable_pd = datatable_pd.drop(columns=del_cols)
    print('Deleted features that were not used by cadd paper.')

    # delete columns without a single value
    datatable_pd = datatable_pd.dropna(axis=1,how='all')
    print('Deleted features without a single row value.')

    # fill in values recommended by cadd paper
    values = {'GerpRS':0, 'GerpRSpval':1,'EncExp':0,'EncOCC':5,
              'EncOCCombPVal':0,'EncOCDNasePVal':0,'EncOCFairePVal':0,
              'EncOCpolIIPVal':0,'EncOCctcfPVal':0,'EncOCmycPVal':0,
              'EncOCDNaseSig':0,'EncOCFaireSig':0,'EncOCpolIISig':0,
              'EncOCctcfSig':0,'EncOCmycSig':0,'tOverlapMotifs':0,
              'motifDist':0,'TFBS':0,'TFBSPeaksMax':0,'PolyPhenVal':0,
              'SIFTval':0,'TFBSPeaks':0}
    datatable_pd = datatable_pd.fillna(values)
    print('Impute with recommended values in cadd paper.')

    # drop nan values -TODO
    print('Deleted columns that I do not know how to impute:')
    for col in datatable_pd.select_dtypes(exclude=[np.object]).columns:
        null = datatable_pd[col].isnull().values.ravel().sum()
        if null > 0:
            print(null,col)
            datatable_pd = datatable_pd.drop(columns=col)

    return datatable_pd


def create_dummy_scale(datatable_pd):
    '''
    create dummy variables
    '''
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
    dummy_data['INFO'] = dummy_data['INFO'].astype('category').cat.codes
    check_null(dummy_data)
    # normalized the numerical values before any processing afterwards
    min_max_scaler = MinMaxScaler()
    dummy_data_scaled = min_max_scaler.fit_transform(dummy_data)
    dummy_data_scaled = pd.DataFrame(dummy_data_scaled,
                                     columns=dummy_data.columns)
    return dummy_data_scaled

def preprocess_dataset(datatable,name,
                       POS=False,
                       GeneName=False,
                       GavinRes=False,
                       SAVE_PATH=None):
    print('The original data table has shape:',datatable.shape)
    datatable.name = name
    if hasattr(datatable,'GeneName'):
        genes = datatable['GeneName'].values
    if hasattr(datatable,'gavin_res'):
        gavin = datatable['gavin_res'].values
        datatable = datatable.drop(['gavin_res'],axis=1)
    if hasattr(datatable,'POS'):
        pos = datatable.POS.values
    if hasattr(datatable,'Pos'):
        pos = datatable.Pos.values

    # select columns to remain/left-out
    if name == 'myh7':
        datatable = drop_impute_full(datatable)
    elif name == 'myo5b':
        datatable = drop_impute_myo5b(datatable)
    elif name == 'myh7_myo5b':
        datatable = drop_impute_myo5b(datatable)

    # create dumy variables and scale to 0-1
    datatable = create_dummy_scale(datatable)

    if POS:
        datatable['POS'] = Pos
        print('Added POS to the table.')

    if GeneName:
        datatable['genename'] = genes
        print('Added genename to the table.')

    if GavinRes:
        datatable['gavin_res'] = gavin
        datatable['gavin_res'] = datatable['gavin_res'].astype('category').cat.codes
        print('Added gavin_res to the table.')

    print('Datatable {0} preprocessed, final shape: {1}.'.format(name,
                                                        datatable.shape))
    if SAVE_PATH:
        # save the preprocessed data as csv file
        datatable.to_csv(SAVE_PATH,sep='\t',index=False)
        print('Saved to path: ',SAVE_PATH)
    return datatable
