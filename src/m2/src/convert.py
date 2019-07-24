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

def cv_convert(pred_path, out_path):
    df_pred = loader.load_df(pred_path)

    df_pred = df_pred.sort_values(['session_id', 'target'], ascending=False)
    df_pred['rank'] = df_pred.groupby('session_id').cumcount()

    df_pred = df_pred.sort_values(by=['session_id', 'rank']).reset_index(drop=True)
    df_pred.to_csv(out_path, float_format='%.4f', index=False)


def sub_convert(df_path, pred_path, out_path):
    df_data = loader.load_df(df_path)
    df_pred = loader.load_df(pred_path)

    required_cols = ['user_id','session_id','timestamp','step','impressions']
    df_sub = df_data[required_cols]
    df_sub['target'] = df_pred['target']
    df_sub = df_sub.sort_values(by=['session_id', 'target'], \
            ascending=False).reset_index(drop=True)

    df_sub['item_recommendations'] = df_sub['impressions'].astype(str)
    df_sub = df_sub.groupby(['user_id','session_id','timestamp','step']) \
                ['item_recommendations'].apply(lambda lst : ' '.join((lst))).reset_index()

    df_sub = df_sub.sort_values(by='session_id').reset_index(drop=True) 
    df_sub.to_csv(out_path, float_format='%.4f', index=False)


if __name__ == "__main__":

    print('start time: %s' % datetime.now())
    root_path = '../feature/'
    base_tr_path = root_path + 'tr_s0_0.ftr'
    base_te_path = root_path + 'te_s0_0.ftr'

    sub_file_path = sys.argv[1]
    sub_name = sys.argv[2]

    cv_path = '{}/{}_cv.csv'.format(sub_file_path, sub_name)
    sub_path = '{}/{}.csv'.format(sub_file_path, sub_name)

    cv_out_path = '{}/r_{}_cv.csv'.format(sub_file_path, sub_name)
    sub_out_path = '{}/r_{}.csv'.format(sub_file_path, sub_name)

    cv_convert(cv_path, cv_out_path)
    sub_convert(base_te_path, sub_path, sub_out_path)

    print('all completed: %s' % datetime.now())

