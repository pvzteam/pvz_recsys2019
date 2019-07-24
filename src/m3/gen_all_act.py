import pandas as pd
import sys
import utils
import config

nrows = None
tr = utils.load_df(config.data+'train.csv',nrows=nrows)
te = utils.load_df(config.data+'test.csv',nrows=nrows)

actions = ['interaction item image','interaction item info','interaction item deals','interaction item rating','search for item'
        ]

df = pd.concat([tr,te])

trs = utils.load_df(config.feat+'m3_tr_0.ftr')
tes = utils.load_df(config.feat+'m3_te_0.ftr')

tr_out = trs[['session_id','impressions']]
te_out = tes[['session_id','impressions']]

for act in actions:
    tmp = df[df.action_type==act][['reference','user_id']]
    tmp = tmp.groupby(['reference'])['user_id'].agg(['count','nunique']).reset_index()
    tmp.rename(columns = {'reference':'impressions','count':act+'_pv','nunique':act+'_uv'},inplace=True)
    tmp['impressions'] = tmp['impressions'].astype(str)
    num_index = tmp['impressions'].str.isnumeric()
    tmp = tmp[num_index]
    tmp['impressions'] = tmp['impressions'].astype('int')    
    print(tmp.head())
    tr_out = tr_out.merge(tmp, on = ['impressions'], how = 'left')
    te_out = te_out.merge(tmp, on = ['impressions'], how = 'left')

tr_out.columns = tr_out.columns.astype(str)
te_out.columns = te_out.columns.astype(str)
utils.save_df(tr_out,config.feat+'m3_tr_item_act_pv.ftr')
utils.save_df(te_out,config.feat+'m3_te_item_act_pv.ftr')

