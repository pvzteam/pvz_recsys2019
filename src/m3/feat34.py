import pandas as pd
import sys
import utils
import gc
import config

def convert(ori,des,feats):
    df_ori = utils.load_df(ori)
    print(df_ori.shape)
    for feat in feats:
        df_feat = utils.load_df(config.feat + feat)
        df_ori = df_ori.merge(df_feat,on=['session_id','impressions'],how='left')
        print(df_ori.shape)
        del df_feat
        gc.collect()
    df_ori = utils.reduce_mem(df_ori)
    df_ori.columns = df_ori.columns.astype(str)
    utils.save_df(df_ori, des)

ori = 32
des = sys.argv[0][4:-3]
tr_feats = ['m2_tr_top30_fea.ftr','m1_top_30_feats_train.ftr']
te_feats = ['m2_te_top30_fea.ftr','m1_top_30_feats_test.ftr']
convert(config.feat+'m3_tr_%d.ftr' % ori,config.feat+'m3_tr_%s.ftr' % des, tr_feats)
convert(config.feat+'m3_te_%d.ftr' % ori,config.feat+'m3_te_%s.ftr' % des, te_feats)
