import pandas as pd
import sys
import utils
import cate_encoding
import config

def convert(df_ori, des,df_out):
    print(df_ori.shape)
    df_ori = df_ori.merge(df_out, on = ['session_id','impressions'], how = 'left')
    df_ori.columns = df_ori.columns.astype(str)
    print(df_ori.head())
    utils.save_df(df_ori, des)

def gen_neighbor(df):
    df['nb_left'] = df.groupby('session_id')['impressions'].shift(1)
    df['nb_right'] = df.groupby('session_id')['impressions'].shift(-1)
    df['price_left_div'] = df['prices'] / df.groupby('session_id')['prices'].shift(1)
    df['price_right_div'] = df['prices'] / df.groupby('session_id')['prices'].shift(-1)
    df_out = df[['session_id','impressions','nb_left','nb_right','price_left_div','price_right_div']] 
    return df_out

nrows = None
trs = utils.load_df(config.feat+'m3_tr_0.ftr')
tes = utils.load_df(config.feat+'m3_te_0.ftr')

convert(trs[['session_id','impressions']],config.feat+'m3_tr_sid_item_neighbor.ftr',gen_neighbor(trs))
convert(tes[['session_id','impressions']],config.feat+'m3_te_sid_item_neighbor.ftr',gen_neighbor(tes))



