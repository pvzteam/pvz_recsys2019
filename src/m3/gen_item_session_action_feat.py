import pandas as pd
import sys
import utils
import config

nrows = None
tr = utils.load_df(config.data+'train.csv',nrows=nrows)
te = utils.load_df(config.data+'test.csv',nrows=nrows)

actions = ['interaction item image','interaction item info','interaction item deals','interaction item rating','search for item']

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
for act in actions:
    feat = df[df.action_type==act][['session_id','reference','step_x']]
    feat = feat.groupby(['session_id','reference'])['step_x'].agg(['count']).reset_index()
    feat.rename(columns = {'reference':'impressions','count':act+'_item_cnt'},inplace=True)
    print(feat.head())
    feat['impressions'] = feat['impressions'].astype(str)
    num_index=feat['impressions'].str.isnumeric()
    feat = feat[num_index]
    feat['impressions'] = feat['impressions'].astype('int')
    tr_out = tr_out.merge(feat, on = ['session_id','impressions'], how = 'left')
    te_out = te_out.merge(feat, on = ['session_id','impressions'], how = 'left')

tr_out.columns = tr_out.columns.astype(str)
te_out.columns = te_out.columns.astype(str)
utils.save_df(tr_out,config.feat+'m3_tr_item_sid_act.ftr')
utils.save_df(te_out,config.feat+'m3_te_item_sid_act.ftr')
