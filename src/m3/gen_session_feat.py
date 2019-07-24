import pandas as pd
import sys
import utils
import cate_encoding 
import config

def gen_hist_feat(df_feat, id_name, cate_name, ratio = False, prefix = ""):
    df_feat['num'] = 1
    df_cnt = df_feat.groupby([id_name])['num'].agg('sum').reset_index()
    df_feat = df_feat.groupby([id_name,cate_name])['num'].agg('sum').reset_index()
    df_feat = df_feat.set_index([id_name,cate_name])[['num']].unstack(level=-1).fillna(0)
    df_feat.columns = df_feat.columns.get_level_values(1)
    df_feat = df_feat.reset_index()
    print(df_feat.head())
    if ratio:
        df_feat = df_feat.merge(df_cnt,on=id_name, how='left')
        for c in df_feat.columns:
            if c == id_name or c == 'num':
                continue
            df_feat[c] = df_feat[c] / df_feat['num']
        df_feat.drop('num',axis=1,inplace=True)
    df_feat.columns = df_feat.columns.astype(str)
    d_name = {}
    for c in df_feat.columns:
        if c == id_name:
            continue
        d_name[c] = prefix + c
    print(d_name)
    df_feat.rename(columns = d_name, inplace=True)
    print(df_feat.head())
    return df_feat

def extract(df_ori, des):
    print(df_ori.shape)
    df_ori = df_ori.merge(df_hist, on = ['session_id'], how = 'left')
    df_ori = df_ori.merge(df_sid, on = ['session_id'], how = 'left')
    print(df_ori.head(10))
    df_ori.columns = df_ori.columns.astype(str)
    utils.save_df(df_ori, des)

nrows = None
tr = utils.load_df(config.data+'train.csv',nrows=nrows)
te = utils.load_df(config.data+'test.csv',nrows=nrows)

df = pd.concat([tr,te])
df_out = df[['session_id']]

trs = utils.load_df(config.feat+'m3_tr_0.ftr')
tes = utils.load_df(config.feat+'m3_te_0.ftr')
df_sample = pd.concat([trs,tes])
df_sample = df_sample[['session_id','step']].drop_duplicates()
df = df.merge(df_sample,on='session_id',how='left')
print(df.head(10))
df = df[df.step_x < df.step_y]

tr_out = trs[['session_id','impressions']]
te_out = tes[['session_id','impressions']]

# sid time gap
df_sid = df.groupby('session_id')['timestamp'].agg(['min','max','count','nunique']).reset_index()
df_sid['sid_duration'] = df_sid['max'] - df_sid['min']
df_sid.drop(['max','min'],axis=1,inplace=True)
df_sid.rename(columns = {'count':'sid_ts_cnt','nunique':'sid_ts_nunique'},inplace=True)

df_hist = gen_hist_feat(df,'session_id','action_type',prefix = "sid_")
print(df_hist.head())

extract(tr_out,config.feat+'m3_tr_sid_feat.ftr')
extract(te_out,config.feat+'m3_te_sid_feat.ftr')


