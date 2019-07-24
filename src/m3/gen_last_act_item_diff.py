import pandas as pd
import sys
import utils
import cate_encoding
import config

def convert(df_ori, des):
    print(df_ori.shape)
    df_ori = df_ori.merge(df_last, on = ['session_id'], how = 'left')
    df_ori['last_item_rank_diff'] = df_ori['impr_rank'] - df_ori['last_item_impr_rank']
    df_ori['last_item_price_div'] = df_ori['prices'] / df_ori['last_item_price']
    df_ori.drop(['last_item_impr_rank','last_item_price','prices','impr_rank'],axis=1,inplace=True)
    df_ori.columns = df_ori.columns.astype(str)
    print(df_ori.head())
    utils.save_df(df_ori, des)

nrows = None
tr = utils.load_df(config.data+'train.csv',nrows=nrows)
te = utils.load_df(config.data+'test.csv',nrows=nrows)
df = pd.concat([tr,te])
df = df[['user_id','session_id','timestamp','step','action_type','reference']]
trs = utils.load_df(config.feat+'m3_tr_0.ftr')
tes = utils.load_df(config.feat+'m3_te_0.ftr')
df_sample = pd.concat([trs,tes])
df_sid = df_sample[['session_id','timestamp']].drop_duplicates(subset=['session_id'])
df = df.merge(df_sid,on='session_id',how='left')
print(df.head(10))
print(df.shape)
mask = (df.timestamp_x < df.timestamp_y) & pd.notnull(df.timestamp_y)
df = df[mask]

df['reference'] = df['reference'].astype(str)
num_index= df['reference'].str.isnumeric()
df = df[num_index]
df.rename(columns={'reference':'impressions'},inplace=True)
df['impressions'] = df['impressions'].astype('int')
df.sort_values(by=['session_id','timestamp_x'],inplace=True)
df_last = df[['session_id','impressions']].drop_duplicates(subset=['session_id'],keep='last')

df_sample = df_sample.drop_duplicates(subset=['session_id','impressions'])
df_last = df_last.merge(df_sample[['session_id','impressions','impr_rank','prices']], on = ['session_id','impressions'], how='left')
print(df_last.shape)
df_last = df_last[pd.notnull(df_last.impr_rank)]
df_last.rename(columns={'impr_rank':'last_item_impr_rank','prices':'last_item_price','impressions':'last_impr'},inplace=True)
print(df_last.shape)

convert(trs[['session_id','impressions','impr_rank','prices']],config.feat+'m3_tr_last_item_diff.ftr')
convert(tes[['session_id','impressions','impr_rank','prices']],config.feat+'m3_te_last_item_diff.ftr')



