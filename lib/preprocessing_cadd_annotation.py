import pandas as pd
import os

def drop_impute(datatable_pd,GeneName=False,POS=False,SAVE_PATH=None):
    print('The original data table has shape:',datatable_pd.shape)
    datatable_pos = datatable_pd['Pos']
    genename = datatable_pd['GeneName'].values
    # delete some columns that were not used in cadd paper
    del_cols = ['#Chrom','Pos','isDerived','AnnoType','ConsScore',
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
    for col in datatable_pd.columns:
        null = datatable_pd[col].isnull().values.ravel().sum()
        if null > 0:
            print(null,col)
            datatable_res = datatable_pd.drop(columns=col)
    print('Deleted columns that I do not know how to impute:')

    if GeneName:
        datatable_pd['key'] = genename
        print('Added a column with GeneName.')
    if POS:
        datatable_pd['POS'] = datatable_pos
        print('Added a column with POS.')
    if SAVE_PATH:
        datatable_pd.to_csv(SAVE_PATH,sep='\t',index=False)
        print('The processed data table is saved to %s'%SAVE_PATH)

    return datatable_pd
