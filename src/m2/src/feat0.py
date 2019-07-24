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
from math import sqrt
from collections import Counter

# 自定义工具包
sys.path.append('../tools/')
import loader
import cate_encoding

# 设置随机种子
SEED = 2018
np.random.seed (SEED)

input_root_path  = '../input/'
output_root_path = '../feature/'

tr_base_path = input_root_path + 'tr.ftr'
te_base_path = input_root_path + 'te.ftr'

cv_id_path = '../../../input/' + 'cvid.csv'

postfix = 's0_0'
file_type = 'ftr'

# 当前特征
tr_fea_out_path = output_root_path + 'tr_fea_{}.{}'.format(postfix, file_type)
te_fea_out_path = output_root_path + 'te_fea_{}.{}'.format(postfix, file_type)

# 当前特征 + 之前特征 merge 之后的完整训练数据
tr_out_path = output_root_path + 'tr_{}.{}'.format(postfix, file_type)
te_out_path = output_root_path + 'te_{}.{}'.format(postfix, file_type)


ID_NAME = ['session_id', 'impressions']
TARGET_NAME = 'target'

def feat_extract(df):
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['dt'].dt.hour
    df.drop(['dt'], axis=1, inplace=True)

    cate_cols = ['city', 'device', 'platform', 'current_filters']
    for col in cate_cols:
        df[col] = pd.factorize(df[col], sort=True)[0]

    # impr rank
    df['impr_rank'] = df.groupby(['session_id']).cumcount().values

    # price statistics by session
    df = cate_encoding.cate_num_stat(df, df, ['session_id'], 'prices', ['median','std','count'])

    df['price_sub'] = df['prices'] - df['session_id_by_prices_median']
    df['price_div'] = df['prices'] / df['session_id_by_prices_median']

    return df

def gen_fea():
    tr = loader.load_df(tr_base_path)
    te = loader.load_df(te_base_path)

    df_base = pd.concat([tr, te])

    df_base = feat_extract(df_base)

    tr = df_base[pd.notnull(df_base['target'])].reset_index(drop=True)
    te = df_base[pd.isnull(df_base['target'])].reset_index(drop=True)

    cv_df = pd.read_csv(cv_id_path)
    cv_df['cv'] += 1
    tr = tr.merge(cv_df, on='session_id', how='left')
    te['cv'] = 0

    print (tr.head())
    print (te.head())

    loader.save_df(tr, tr_out_path)
    loader.save_df(te, te_out_path)


if __name__ == "__main__":

    print('start time: %s' % datetime.now())

    gen_fea()

    print('all completed: %s' % datetime.now())

