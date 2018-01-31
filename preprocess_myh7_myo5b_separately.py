'''
preprocess myh7.csv and myo5b*.xlxs separately
select features
create dummy variables
scale numerical values to 0-1
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

    # process the myh7 gene dataset
    myh7 = os.path.join('data','myh7','myh7.csv')
    myh7_pd = pd.read_csv(myh7,sep='\t')
    save_path = os.path.join('data','myh7','myh7_dummy_no_nan_data.csv')
    myh7_pd = preprocess_dataset(myh7_pd,'myh7',SAVE_PATH=save_path)

    # preprocess the myo5b Dataset
    myo5b = os.path.join('data','myo5b',
                 'myo5b_variants_patho_benign_cadd1.3fullannot_v1.xlsx')
    myo5b_excel = pd.ExcelFile(myo5b)
    myo5b_pd = myo5b_excel.parse(myo5b_excel.sheet_names[0])
    del myo5b_excel
    save_path = os.path.join('data','myo5b','myo5b_dummy_no_nan.csv')
    myo5b_pd = preprocess_dataset(myo5b_pd,'myo5b',SAVE_PATH=save_path)
