import pandas as pd
import numpy as np
import sys
import utils
import config
def gen_train_sample(df):
    df['target'] = (df['reference'] == df['impressions']).astype(int)
    df.drop(['current_filters','reference','action_type'],axis=1,inplace=True)
    df_session = df[['session_id','step']].drop_duplicates(subset='session_id',keep='last').reset_index(drop=True)
    df = df_session.merge(df, on=['session_id','step'], how='left').reset_index(drop=True)
    #loader.save_df(df,config.data+'m3_tr.ftr')
    return df

def get_test_sample(df):
    df['target'] = (df['reference'] == df['impressions']).astype(int)
    # drop noisy sample
    mask = (df.session_id == 'cbe3752713eee') & (df.timestamp ==1541660358)
    df = df[~mask]
    df_session = df[['session_id','step']].drop_duplicates(subset='session_id',keep='last').reset_index(drop=True)
    df = df_session.merge(df, on=['session_id','step'], how='left').reset_index(drop=True)
    te = df[pd.isnull(df['reference'])].reset_index(drop=True)
    print(te.shape)
    tr = df[pd.notnull(df['reference'])].reset_index(drop=True)
    print(tr.shape) 
    tr.drop(['current_filters','reference','action_type'],axis=1,inplace=True)
    te.drop(['current_filters','reference','action_type','target'],axis=1,inplace=True)
    utils.save_df(te,config.data+'m3_te.ftr')
    return tr,te

def gen_tr_click(df):
    df = df[['session_id','reference']].drop_duplicates(subset='session_id',keep='last').reset_index(drop=True)
    print(df.shape)
    df = df[pd.notnull(df.reference)].reset_index(drop=True)
    print(df.shape)
    utils.save_df(df,config.data+'m3_tr_click.ftr')

if __name__ == '__main__':
    nrow = None
    train = utils.load_df(config.data+'sample_train.csv',nrows=nrow)
    test = utils.load_df(config.data+'sample_test.csv',nrows=nrow)
    df = pd.concat([train,test]).reset_index(drop=True)
    tr1 = gen_train_sample(train)
    tr2,te = get_test_sample(test)
    tr = pd.concat([tr1,tr2]).reset_index(drop=True)
    utils.save_df(tr1,config.data+'m3_tr.ftr')
    gen_tr_click(df)
