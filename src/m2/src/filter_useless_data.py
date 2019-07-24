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

def remove_acts_after_last_clk(df, is_te=False):
    print ('remove_acts_after_last_clk')
    print ('df shape', df.shape)
    df_sample = df[df.action_type == 'clickout item']

    if is_te == False:
        df_sample = df_sample[['session_id', 'timestamp']] \
                .drop_duplicates(subset='session_id', keep='last') \
                .reset_index(drop=True)
    else:
        df_sample = df_sample[pd.isnull(df_sample.reference)] \
                .reset_index(drop=True)
        df_sample = df_sample[['session_id', 'timestamp']]
    df_sample.columns = ['session_id', 'timestamp_max']

    df = df.merge(df_sample, on='session_id', how='left')
    print(df.head(10))
    df = df[df.timestamp <= df.timestamp_max]
    del df['timestamp_max']
    print ('df shape', df.shape)

    return df

def remove_repeated_session_in_tr(tr, te):
    print ('remove_repeated_session_in_tr')
    tr_se = set(tr.session_id.tolist())
    te_se = set(te.session_id.tolist())
    commons = list(tr_se & te_se)

    print ('tr shape', tr.shape)
    tr = tr[~tr.session_id.isin(commons)]
    print ('tr shape', tr.shape)

    return tr

def remove_invalid_reference(df_ori):
    print ('remove_invalid_reference')
    print ('df shape', df_ori.shape)
    required_cols = ['session_id', 'action_type', 'reference']
    df = df_ori[required_cols]

    actions = ['interaction item image', 'interaction item info', \
            'interaction item deals', 'interaction item rating', \
            'search foritem']
    df = df[df.action_type.isin(actions)]

    # 过滤掉 reference 中非数值的数据
    df['reference'] = df['reference'].astype(str)
    num_index = df['reference'].str.isnumeric()

    mask = (~df_ori.action_type.isin(actions)) | \
            (df_ori.action_type.isin(actions) & num_index)
    df_ori = df_ori[mask]
    print ('df shape', df_ori.shape)

    return df_ori


def filter_useless_data():

    tr = loader.load_df('../input/train.ftr')
    te = loader.load_df('../input/test.ftr')

    tr = remove_repeated_session_in_tr(tr, te)

    tr = remove_invalid_reference(tr)
    te = remove_invalid_reference(te)

    tr = remove_acts_after_last_clk(tr, is_te=False)
    te = remove_acts_after_last_clk(te, is_te=True)

    loader.save_df(tr, '../input/train.ftr')
    loader.save_df(te, '../input/test.ftr')


if __name__ == "__main__":

    print('start time: %s' % datetime.now())
    tr = loader.load_df('../../../input/train.csv')
    te = loader.load_df('../../../input/test.csv')
    item_meta = loader.load_df('../../../input/item_metadata.csv')

    loader.save_df(tr, '../input/train.ftr')
    loader.save_df(te, '../input/test.ftr')
    loader.save_df(item_meta, '../input/item_metadata.ftr')

    filter_useless_data()

    print('all completed: %s' % datetime.now())

