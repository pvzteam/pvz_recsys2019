import pandas as pd
import sys
import utils
import config

def convert(ori, des, feats):
    df_ori = utils.load_df(ori)
    for f in feats:
        tmp = utils.load_df(config.feat+'m3_' +f)
        print(f)
        df_ori = pd.concat([df_ori,tmp.drop(['session_id','impressions'],axis=1)],axis=1)
    df_ori.columns = df_ori.columns.astype(str)
    utils.save_df(df_ori,des)

tr_cols = ['tr_item_sid_act.ftr',
           'tr_item_sid_clk_impr.ftr',
           'tr_item_act_pv.ftr',
           'tr_ctr.ftr',
           'tr_item_last_act.ftr',
           'tr_sid_feat.ftr',
           'tr_sid_last_act_feat.ftr',
           'tr_item_feat.ftr']
te_cols = [c.replace('tr_','te_') for c in tr_cols]
print(te_cols)
ori = 0
des = sys.argv[0][4:-3]
convert(config.feat+'m3_tr_%d.ftr' % ori,config.feat+'m3_tr_%s.ftr' % des, tr_cols)
convert(config.feat+'m3_te_%d.ftr' % ori,config.feat+'m3_te_%s.ftr' % des, te_cols)
