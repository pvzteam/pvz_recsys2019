import pandas as pd
import sys
import utils
import cate_encoding 
import config

def convert(df_ori, des):
    print(df_ori.shape)
    df_ori = df_ori.merge(df_out, on = ['session_id','impressions'], how = 'left')
    df_ori['cur_ts_sub_user_last'] = df_ori['timestamp'] - df_ori['last_ts']
    df_ori.drop(['timestamp','last_ts'],axis=1,inplace=True)
    df_ori.columns = df_ori.columns.astype(str)
    utils.save_df(df_ori, des)

nrows = None
tr = utils.load_df(config.data+'train.csv',nrows=nrows)
te = utils.load_df(config.data+'test.csv',nrows=nrows)
df = pd.concat([tr,te])
df = df[['user_id','session_id','timestamp','step','action_type','reference']]
trs = utils.load_df(config.feat+'m3_tr_0.ftr')
tes = utils.load_df(config.feat+'m3_te_0.ftr')
df_sample = pd.concat([trs,tes])
df_sid = df_sample[['session_id','timestamp']].drop_duplicates()
df = df.merge(df_sid,on='session_id',how='left')
print(df.head(10))
print(df.shape)
mask = (df.timestamp_x <= df.timestamp_y) | pd.isnull(df.timestamp_y)
df = df[mask]

df['reference'] = df['reference'].astype(str)
num_index= df['reference'].str.isnumeric()
df = df[num_index]
df.rename(columns={'reference':'impressions'},inplace=True)
df['impressions'] = df['impressions'].astype('int')
df.rename(columns={'timestamp_x':'timestamp'},inplace=True)
df.sort_values(by=['user_id','timestamp'],inplace=True)
df = df[['user_id','session_id','timestamp','action_type','impressions']].drop_duplicates(subset=['session_id','impressions'],keep='last')
print(df.head())

df_sample = df_sample[['user_id','session_id','timestamp','impressions']]
df_sample['action_type'] = 'impr'
df_all = pd.concat([df,df_sample]).reset_index(drop=True)
df_all.sort_values(by=['user_id','timestamp'],inplace=True)
df_feats = []
for i in range(7):
    print(i)
    df_all['last_act'] = df_all.groupby(['user_id','impressions'])['action_type'].shift(1+i)
    df_all['last_sid'] = df_all.groupby(['user_id','impressions'])['session_id'].shift(1+i)
    df_all['last_ts'] = df_all.groupby(['user_id','impressions'])['timestamp'].shift(1+i)
    mask = (pd.notnull(df_all['last_sid'])) & (df_all['last_sid']!=df_all['session_id']) & (df_all['last_act'] != 'impr') & (df_all['action_type']=='impr')
    df_feat = df_all[mask]
    print(df_feat.shape)
    df_feats.append(df_all[mask])

df_out = pd.concat(df_feats).reset_index(drop=True)
print(df_out.shape)
df_out.columns = df_out.columns.astype(str)
#utils.save_df(df_out, config.feat+'user_item_last_act.ftr')
#df_out = loader.load_df('../feature/user_item_last_act_debug.ftr')
df_out = cate_encoding.label_encode(df_out,'last_act')
df_out.rename(columns={'last_act':'user_item_last_act'},inplace=True)
df_out.sort_values(by=['session_id','impressions','last_ts'],inplace=True)
df_out.drop_duplicates(subset = ['session_id','impressions'], keep = 'last', inplace=True)
print(df_out.shape)
df_out = df_out[['session_id','impressions','last_ts','user_item_last_act']]


convert(trs[['session_id','impressions','timestamp']],config.feat+'m3_tr_item_uid_last_act.ftr')
convert(tes[['session_id','impressions','timestamp']],config.feat+'m3_te_item_uid_last_act.ftr')


