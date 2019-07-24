import pandas as pd
import sys
import utils
import cate_encoding 
import config

def extract(df_ori, des):
    print(df_ori.shape)
    df_ori = df_ori.merge(df, on = ['session_id'], how = 'left')
    df_ori['last_act_gap'] = df_ori['timestamp'] - df_ori['timestamp_x']
    df_ori.drop(['timestamp','timestamp_x'],axis=1,inplace=True)
    print(df_ori.head(10))
    df_ori.columns = df_ori.columns.astype(str)
    utils.save_df(df_ori, des)

nrows = None
tr = utils.load_df(config.data+'train.csv',nrows=nrows)
te = utils.load_df(config.data+'test.csv',nrows=nrows)

df = pd.concat([tr,te])
df = df[['session_id','timestamp','step','action_type','reference']]
trs = utils.load_df(config.feat+'m3_tr_0.ftr')
tes = utils.load_df(config.feat+'m3_te_0.ftr')
df_sample = pd.concat([trs,tes])
df_sample = df_sample[['session_id','timestamp']].drop_duplicates()
df = df.merge(df_sample,on='session_id',how='left')
print(df.head(10))
df = df[df.timestamp_x < df.timestamp_y]

df = df[['session_id','timestamp_x','action_type']].drop_duplicates(subset=['session_id'],keep='last')
df = cate_encoding.label_encode(df,'action_type')
df.rename(columns={'action_type':'sid_last_act'},inplace=True)

tr_out = trs[['session_id','impressions','timestamp']]
te_out = tes[['session_id','impressions','timestamp']]
extract(tr_out,config.feat+'m3_tr_sid_last_act_feat.ftr')
extract(te_out,config.feat+'m3_te_sid_last_act_feat.ftr')



