#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 基础模块
import os
import sys
import gc
import json
import time
import functools
from datetime import datetime

# 数据处理
import numpy as np
import pandas as pd

# 自定义工具包
sys.path.append('../tools/')
import loader

# 设置随机种子
SEED = 2018
np.random.seed (SEED)

if __name__ == "__main__":

    print('start time: %s' % datetime.now())
    # tr = loader.load_df('../feature/tr_s0_106.ftr')
    # te = loader.load_df('../feature/te_s0_106.ftr')
    tr = loader.load_df('../../../feat/m2_tr_s0_106.ftr')
    te = loader.load_df('../../../feat/m2_te_s0_106.ftr')

    cols = ['session_id', 'impressions', 'impr_rank', 'ts_sub_prev', 'abs_impr_rank_sub_lastest_item-impr_rank', 'impr_rank_sub_lastest_item-impr_rank', 'nearest_step_delta', 'prices_div_active_items-session_id_by_prices_median-v2', 'price_rank', 'act_pre1', 'lastest_item-impr_rank', 'impr_rank_sub_impressions_by_impr_rank_median', 'price_div', 'session_act_sum', 'prices_div_active_items-session_id_by_prices_median', 'impressions_by_hist_interaction item info_sum', 'impressions_by_hist_interaction item image_sum', 'impressions_by_hist_clickout item_sum', 'session_id_by_prices_count', 'impressions_target', 'impressions_active_ratio', 'price_div_impr_rank_1_price', 'impressions_target_sub_session_median', 'impressions_target_sub_session_max', 'session_hist_clickout item', 'device', 'impr_rank_sub_session_id_by_impr_rank_median', 'session_hist_interaction item image', 'impr_rank_1_impressions_target', 'impr_rank_sub_session_id_by_impr_rank_max', 'impr_rank_sub_session_id_by_impr_rank_min', 'current_filters']

    tr = tr[cols]
    te = te[cols]

    tr.columns = ['session_id', 'impressions'] + \
            ['m2_{}'.format(c) for c in tr.columns.tolist()[2:]]
    te.columns = ['session_id', 'impressions'] + \
            ['m2_{}'.format(c) for c in te.columns.tolist()[2:]]

    loader.save_df(tr, '../../../feat/m2_tr_top30_fea.ftr')
    loader.save_df(te, '../../../feat/m2_te_top30_fea.ftr')

    print('all completed: %s' % datetime.now())

