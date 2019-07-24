import pandas as pd
import sys
import utils
import config

def dump_feat(ori,des):
    df = utils.load_df(ori)
    df = df[cols+['session_id','impressions']]
    df.columns = re_cols + ['session_id','impressions']
    print(df.shape)
    utils.save_df(df,des)

cols = ['session_id_by_last_ts_sub_max_rank', 'impr_rank', 'cur_ts_sub_last', 'last_act_gap', 'last_item_rank_diff', 'last_ts_sub_max', 'session_id_by_prices_rank', 'ctr', 'last_item_price_div', 'item_last_act', 'nb_left', 'sid_last_act', 'price_left_div', 'session_id_impressions_by_impr_rank_min', 'search for item_item_cnt', 'price_div', 'user_item_last_act', 'session_id_by_prices_count', 'interaction item image_item_cnt', 'item_rank_sub_median', 'sid_search for item', 'sid_interaction item image', 'item_all_impr', 'item_sid_impr_cnt', 'price_right_div', 'device', 'item_sid_clk_cnt', 'price_sub', 'nb_right', 'item_price_div_median']

re_cols = ['m3_'+c for c in cols ]

dump_feat(config.feat+'m3_tr_21.ftr',config.feat+'m3_tr_feat_top30.ftr')
dump_feat(config.feat+'m3_te_21.ftr',config.feat+'m3_te_feat_top30.ftr')
