import pandas as pd
import sys
import utils
import config

def convert(ori,des,prefix):
    df_ori = utils.load_df(ori)
    print(df_ori.shape)
    for feat in feats:
        df_feat = utils.load_df(config.model+prefix+'%s.csv' % feat).rename(columns={'target':feat})
        df_ori = df_ori.merge(df_feat[['session_id','impressions',feat]],on=['session_id','impressions'],how='left')
        print(df_ori.shape)
    df_ori.columns = df_ori.columns.astype(str)
    utils.save_df(df_ori, des)

ori = 34
des = sys.argv[0][4:-3]
feats = ['m1_20190622','m1_20190624','m1_20190626','m2_38','m2_87','m2_107']
convert(config.feat+'m3_tr_%d.ftr' % ori, config.feat+'m3_tr_%s.ftr' % des, 'tr_')
convert(config.feat+'m3_te_%d.ftr' % ori, config.feat+'m3_te_%s.ftr' % des, 'te_')
