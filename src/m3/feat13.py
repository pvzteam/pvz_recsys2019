import pandas as pd
import sys
import utils
import cate_encoding
import config

cols = ['prices','ctr','last_ts_sub_max']

def convert(ori, des, feat):
    df_ori = utils.load_df(ori)
    print(df_ori.shape)
    for c in cols:
        df_ori = cate_encoding.cate_num_rank(df_ori, ['session_id'], c, ascending=True, show_agg = True)
    df_ori = df_ori.reset_index(drop=True)
    df_ori.columns = df_ori.columns.astype(str)
    utils.save_df(df_ori, des)
    utils.save_df( df_ori[['session_id','impressions','session_id_by_prices_rank','session_id_by_ctr_rank','session_id_by_last_ts_sub_max_rank']],feat)

ori = 12
des = sys.argv[0][4:-3]
convert(config.feat+'m3_tr_%d.ftr' % ori,config.feat+'m3_tr_%s.ftr' % des, config.feat+'m3_tr_feat_rank.ftr')
convert(config.feat+'m3_te_%d.ftr' % ori,config.feat+'m3_te_%s.ftr' % des, config.feat+'m3_te_feat_rank.ftr')
