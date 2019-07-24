import pandas as pd
import sys
import utils
import config

def extract(sample, ori, feat):
    nrows = None
    df = pd.read_csv(sample,nrows=nrows, usecols = ['session_id','step','reference','impressions'])
    print(df.head())
    df_ori = utils.load_df(ori)
    print(df_ori.head())
    df = df.merge( df_ori[['session_id','step']].drop_duplicates() ,on='session_id',how='left')
    print(df.head())
    df = df[df.step_x < df.step_y]
    
    tmp = df.drop_duplicates(subset=['session_id','step_x'])
    df_clk = tmp.groupby(['session_id','reference'])['step_x'].agg('count').reset_index() 
    print(df_clk.head())
    df_clk.rename(columns = {'reference':'impressions','step_x':'item_sid_clk_cnt'},inplace=True)
    df_impr = df.groupby(['session_id','impressions'])['step_x'].agg('count').reset_index()
    print(df_impr.head())
    df_impr.rename(columns = {'step_x':'item_sid_impr_cnt'},inplace=True)

    df_out = df_ori[['session_id','impressions']]
    df_out = df_out.merge(df_clk,on = ['session_id','impressions'], how = 'left')
    df_out = df_out.merge(df_impr,on = ['session_id','impressions'], how = 'left')
    print(df_out.head())
    df_out.columns = df_out.columns.astype(str)
    utils.save_df(df_out,feat)

if __name__ == '__main__':
    extract(config.data+'sample_test.csv',config.feat+'m3_te_0.ftr',config.feat+'m3_te_item_sid_clk_impr.ftr')
    extract(config.data+'sample_train.csv',config.feat+'m3_tr_0.ftr',config.feat+'m3_tr_item_sid_clk_impr.ftr')
