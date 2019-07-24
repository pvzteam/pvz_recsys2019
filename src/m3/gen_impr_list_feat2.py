import pandas as pd
import sys
import utils
import cate_encoding
import config

def extract_list_cnt(df):
    df['impressions'] = df['impressions'] + '||||||'
    for i in range(5):
        df['rank_%d' % i] = df['impressions'].map(lambda x:x.split('|')[i])
    df_sid = df.drop_duplicates(subset=['session_id'],keep='last')
    cols = []
    for i in range(5):
        print(i)
        df_cnt = df.groupby(['rank_%d' % i])['session_id'].agg(['count','nunique']).reset_index()
        df_cnt.rename(columns={'count':'rank_%d_cnt2' % i,'nunique':'rank_%d_uniq2' % i},inplace=True)
        cols = cols + ['rank_%d_cnt2' % i,'rank_%d_uniq2' % i]
        df_sid = df_sid.merge(df_cnt, on=['rank_%d' % i], how='left')
        #print(df_sid.head())
    return df_sid[cols+['session_id']]

def extract(df_ori, des):
    print(df_ori.shape)
    df_ori = df_ori.merge(df_sid, on = ['session_id'], how = 'left')
    df_ori.columns = df_ori.columns.astype(str)
    utils.save_df(df_ori, des)

rows = None
tr = pd.read_csv(config.data+'train.csv',usecols=['session_id','action_type','reference','impressions'],nrows=rows)
print(tr.shape)
te = pd.read_csv(config.data+'test.csv',usecols=['session_id','action_type','reference','impressions'],nrows=rows)
print(te.shape)
df = pd.concat([tr,te])
print(df.shape)

df = df[df.action_type=='clickout item']
df_sid = extract_list_cnt(df)
print(df_sid.head())

trs = utils.load_df(config.feat+'m3_tr_0.ftr')
tes = utils.load_df(config.feat+'m3_te_0.ftr')
tr_out = trs[['session_id','impressions']]
te_out = tes[['session_id','impressions']]
extract(tr_out, config.feat+'m3_tr_imprlist_feat2.ftr')
extract(te_out, config.feat+'m3_te_imprlist_feat2.ftr')

