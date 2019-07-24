import pandas as pd
import numpy as np
import sys
import utils
import config

tr = pd.read_csv(config.data+'sample_train.csv',usecols = ['session_id','step','reference','impressions','user_id'])
te = pd.read_csv(config.data+'sample_test.csv',usecols = ['session_id','step','reference','impressions','user_id'])

df = pd.concat([tr,te])
print(df.shape)

df = df[pd.notnull(df.reference)]
print(df.shape)

df['click'] = (df['impressions']==df['reference']).astype(int)

cvid = pd.read_csv(config.data+'cvid.csv')
df = df.merge(cvid, on='session_id',how='left')

te_ctr = df.groupby('impressions')['click'].agg(['sum','count']).reset_index()

tr_lis = []
for cv in range(5):
    mask = (df.cv == cv)
    val_tr = df.loc[mask][['session_id','impressions']].drop_duplicates()
    tra_tr = df.loc[~mask]
    tmp = tra_tr.groupby('impressions')['click'].agg(['sum','count']).reset_index()
    val_tr = val_tr.merge(tmp, on='impressions', how = 'left')
    tr_lis.append(val_tr)

tr_ctr = pd.concat(tr_lis,axis=0).reset_index(drop=True)
tr_ctr['ctr'] = tr_ctr['sum'] / tr_ctr['count']
te_ctr['ctr'] = te_ctr['sum'] / te_ctr['count']

trs = utils.load_df(config.feat+'m3_tr_0.ftr')
tes = utils.load_df(config.feat+'m3_te_0.ftr')

tr_out = trs[['session_id','impressions']]
te_out = tes[['session_id','impressions']]

te_out = te_out.merge(te_ctr.drop(['sum','count'],axis=1), on = ['impressions'], how = 'left')
tr_out = tr_out.merge(tr_ctr.drop(['sum','count'],axis=1), on = ['session_id','impressions'], how = 'left')

tr_out.columns = tr_out.columns.astype(str)
te_out.columns = te_out.columns.astype(str)
utils.save_df(tr_out,config.feat+'m3_tr_ctr.ftr')
utils.save_df(te_out,config.feat+'m3_te_ctr.ftr')




