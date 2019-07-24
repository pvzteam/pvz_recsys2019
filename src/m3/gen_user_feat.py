import pandas as pd
import sys
import utils
import cate_encoding
import config

def gen_user_stat_feat():
    rows = None
    tr = pd.read_csv(config.data+'sample_train.csv',nrows=rows)
    te = pd.read_csv(config.data+'sample_test.csv',nrows=rows)
    df = pd.concat([tr,te])
    print(df.shape)
    df_uid = df[['user_id']].drop_duplicates()
    df_item = df[['impressions']].drop_duplicates()
    df_session = df[['session_id']].drop_duplicates()
    # user id
    df_uid = cate_encoding.cate_num_stat(df,df_uid,['user_id'],'city',['nunique'])
    df_uid = cate_encoding.cate_num_stat(df,df_uid,['user_id'],'session_id',['nunique'])
    print(df_uid.head())

    # session 
    df_tmp = df.drop_duplicates(subset=['session_id','impressions'],keep='last')
    df_session = cate_encoding.cate_num_stat(df_tmp,df_session,['session_id'],'impressions',['nunique','count'])
    df_session = cate_encoding.cate_num_stat(df_tmp,df_session,['session_id'],'prices',['mean','max','min'])
    df_session = cate_encoding.cate_num_stat(df_tmp,df_session,['session_id'],'step',['max'])
    df_session['overlap_ratio'] = df_session['session_id_by_impressions_nunique'] / df_session['session_id_by_impressions_count']
    print(df_session.head())

    # user session by mean price and item count
    df_tmp = df.drop_duplicates(subset=['session_id','user_id'])
    df_tmp = df_tmp.merge(df_session,on='session_id',how='left')
    df_uid = cate_encoding.cate_num_stat(df_tmp,df_uid,['user_id'],'session_id_by_impressions_nunique',['median'])
    df_uid = cate_encoding.cate_num_stat(df_tmp,df_uid,['user_id'],'session_id_by_step_max',['median'])
    df_uid = cate_encoding.cate_num_stat(df_tmp,df_uid,['user_id'],'session_id_by_prices_mean',['median'])
    print(df_uid.head())
    return df_uid,df_session

def extract(df_ori, des):
    print(df_ori.shape)
    df_ori = df_ori.merge(df_uid, on = ['user_id'], how = 'left')
    df_ori = df_ori.merge(df_session, on = ['session_id'], how = 'left')
    df_ori.drop('user_id',axis=1,inplace=True)
    print(df_ori.head())
    df_ori.columns = df_ori.columns.astype(str)
    utils.save_df(df_ori, des)

df_uid,df_session = gen_user_stat_feat()
trs = utils.load_df(config.feat+'m3_tr_0.ftr')
tes = utils.load_df(config.feat+'m3_te_0.ftr')
tr_out = trs[['session_id','user_id','impressions']]
te_out = tes[['session_id','user_id','impressions']]
extract(tr_out, config.feat+'m3_tr_user_feat.ftr')
extract(te_out, config.feat+'m3_te_user_feat.ftr')

