import pandas as pd
import sys
import utils
import cate_encoding
import config

def convert(ori, des, sample):
    tr = utils.load_df(ori)
    print(tr.shape)
    tr_out = tr[['session_id','impressions']]
    dfs = utils.load_df(sample)
    dfs['impr_rank'] = dfs.groupby(['session_id','step']).cumcount().values
    print(dfs.head())
    tr_out = cate_encoding.cate_num_stat(dfs,tr_out,['session_id','impressions'],'impr_rank',['min','max','median'])
    tr_out.columns = tr_out.columns.astype(str)
    print(tr_out.head())
    utils.save_df(tr_out,des)

if __name__ == '__main__':
    convert(config.feat+'m3_tr_0.ftr',config.feat+'m3_tr_sid_impr_rank.ftr',config.data+'sample_train.csv')
    convert(config.feat+'m3_te_0.ftr',config.feat+'m3_te_sid_impr_rank.ftr',config.data+'sample_test.csv')
