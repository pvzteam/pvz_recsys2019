import pandas as pd
import sys
import utils
import config
import cate_encoding

tr = utils.load_df(config.data+'m3_tr.ftr')
te = utils.load_df(config.data+'m3_te.ftr')
df = pd.concat([tr,te]).reset_index(drop=True)
df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
df['hour'] = df['dt'].dt.hour

cols = ['city','device','platform']
for c in cols:
    df = cate_encoding.label_encode(df, c)

# impr rank 
df['impr_rank'] = df.groupby(['session_id']).cumcount().values
# price statistics by session
df = cate_encoding.cate_num_stat(df,df,['session_id'],'prices',['median','std','count'])

df['price_sub'] = df['prices'] - df['session_id_by_prices_median']
df['price_div'] = df['prices'] / df['session_id_by_prices_median']
df.drop(['dt'],axis=1,inplace=True)
df.columns = df.columns.astype(str)

utils.save_df(df[pd.isnull(df['target'])].reset_index(drop=True), config.feat+'m3_te_0.ftr')
utils.save_df(df[pd.notnull(df['target'])].reset_index(drop=True),config.feat+'m3_tr_0.ftr')
