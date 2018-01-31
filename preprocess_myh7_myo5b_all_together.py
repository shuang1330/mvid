'''
preprocess myh7.csv and myo5b*.xlxs all together
select features
create dummy variables
scale numerical values to 0-1
add gavin benchmark to the table
'''

import pandas as pd
import os
from os.path import isfile, join
import numpy as np
# from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from lib.preprocessing_cadd_annotation import *
from lib.read_data import check_null

if __name__=='__main__':
    # process the myh7 gene dataset so that it has the same columns as myo5b
    myh7 = os.path.join('data','myh7','myh7.csv')
    myh7_pd = pd.read_csv(myh7,sep='\t')
    myh7_pd = myh7_pd.rename(index=str,
                          columns={'#Chrom':'CHROM',
                                   'Pos':'POS',
                                   'Ref':'REF',
                                   'Alt':'ALT'})
    myh7_pd = myh7_pd.drop(['Anc','chr_pos'],axis=1)
    myh7_pd['ID'] = np.nan
    myh7_pd.loc[myh7_pd['INFO']=='POPULATION','INFO']='Benign'
    myh7_pd.loc[myh7_pd['INFO']=='PATHOGENIC','INFO']='Pathogenic'
    # load gavin_res to the myh7
    myh7_gavin_pd = pd.read_csv('data/myh7/mhy7_gavin_pred.csv',sep='\t')
    myh7_gavin_dic = myh7_gavin_pd.set_index('POS')['Classification'].to_dict()
    myh7_pd['gavin_res'] = myh7_pd['POS'].apply(lambda row:myh7_gavin_dic[row])
    del myh7_gavin_dic,myh7_gavin_pd


    # load the myo5b Dataset
    myo5b = os.path.join('data','myo5b',
                 'myo5b_variants_patho_benign_cadd1.3fullannot_v1.xlsx')
    myo5b_excel = pd.ExcelFile(myo5b)
    myo5b_pd = myo5b_excel.parse(myo5b_excel.sheet_names[0])
    # load gavin to myo5b
    myo5b_gavin_pd = pd.read_csv('data/myo5b/myo5b_gavin_res.csv',sep='\t')
    myo5b_gavin_dic = myo5b_gavin_pd.set_index('POS')['Classification'].to_dict()
    myo5b_pd['gavin_res'] = myo5b_pd['POS'].apply(lambda row:myo5b_gavin_dic[row])
    del myo5b_excel,myo5b_gavin_pd,myo5b_gavin_dic

    # concatenate the myh7 and myo5b datasets
    print('Differences:')
    print(set(myh7_pd.columns).symmetric_difference(set(myo5b_pd.columns)))
    myh7_myo5b = pd.concat([myh7_pd,myo5b_pd],axis=0)

    # preprocess the concatenated dataset
    save_path = os.path.join('data','all_variants',
                             'myh7_myo5b_dummy_no_nan_with_gavin.csv')

    print(myh7_myo5b[['INFO','gavin_res']].head())
    myh7_myo5b = preprocess_dataset(myh7_myo5b,
                                'myh7_myo5b',
                                GeneName=True,
                                GavinRes=True,
                                SAVE_PATH=save_path)
    print(myh7_myo5b[['INFO','gavin_res']].head())
