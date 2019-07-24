import pandas as pd
import sys
import utils
import cate_encoding 
import config

def convert(df_ori, des):
    print(df_ori.shape)
    df_ori = df_ori.merge(df, on = ['session_id','impressions'], how = 'left')
    df_ori['cur_ts_sub_last'] = df_ori['timestamp'] - df_ori['timestamp_x']
    df_ori.drop(['timestamp','timestamp_x'],axis=1,inplace=True)
    df_ori.columns = df_ori.columns.astype(str)
    utils.save_df(df_ori, des)
 

nrows = None
tr = utils.load_df(config.data+'train.csv',nrows=nrows)
te = utils.load_df(config.data+'test.csv',nrows=nrows)

actions = ['interaction item image','interaction item info',
           'interaction item deals','interaction item rating','search for item']
df = pd.concat([tr,te])
df = df[['session_id','timestamp','step','action_type','reference']]
trs = utils.load_df(config.feat+'m3_tr_0.ftr')
tes = utils.load_df(config.feat+'m3_te_0.ftr')
df_sample = pd.concat([trs,tes])
df_sample = df_sample[['session_id','step']].drop_duplicates()
df = df.merge(df_sample,on='session_id',how='left')
print(df.head(10))
df = df[df.step_x < df.step_y]

df['reference'] = df['reference'].astype(str)
num_index= df['reference'].str.isnumeric()
df = df[num_index]
df.rename(columns={'reference':'impressions'},inplace=True)
df['impressions'] = df['impressions'].astype('int')

df = df[['session_id','timestamp','action_type','impressions']].drop_duplicates(subset=['session_id','impressions'],keep='last')
df = cate_encoding.cate_num_stat(df,df,['session_id'],'timestamp','max')
df['last_ts_sub_max'] = df['timestamp_y'] - df['timestamp_x']
df.drop('timestamp_y',axis=1,inplace=True)

df = cate_encoding.label_encode(df,'action_type')
df.rename(columns={'action_type':'item_last_act'},inplace=True)
print(df.head())

convert(trs[['session_id','impressions','timestamp']],config.feat+'m3_tr_item_last_act.ftr')
convert(tes[['session_id','impressions','timestamp']],config.feat+'m3_te_item_last_act.ftr')


