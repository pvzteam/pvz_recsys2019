import pandas as pd
import sys
import utils
import config

def convert(ori,des,feat):
    df_ori = utils.load_df(ori)
    print(df_ori.shape)
    df_feat = utils.load_df(config.feat + feat)
    df_ori = df_ori.merge(df_feat,on=['session_id','impressions'],how='left')
    print(df_ori.shape)
    df_ori.columns = df_ori.columns.astype(str)
    utils.save_df(df_ori, des)

ori = 71
des = sys.argv[0][4:-3]
convert(config.feat+'m3_tr_%d.ftr' % ori, config.feat+'m3_tr_%s.ftr' % des, 'm2_tr_pairwise_fea_s0_108.ftr')
convert(config.feat+'m3_te_%d.ftr' % ori, config.feat+'m3_te_%s.ftr' % des, 'm2_te_pairwise_fea_s0_108.ftr')
