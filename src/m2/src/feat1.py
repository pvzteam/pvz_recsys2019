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
import custom_cate_encoding

# 设置随机种子
SEED = 2018
np.random.seed (SEED)

input_root_path = '../input/'
output_root_path = '../feature/'

tr_base_path = input_root_path + 'train.ftr'
te_base_path = input_root_path + 'test.ftr'

postfix = 's0_1'
file_type = 'ftr'

# 当前特征
tr_fea_out_path = output_root_path + 'tr_fea_{}.{}'.format(postfix, file_type)
te_fea_out_path = output_root_path + 'te_fea_{}.{}'.format(postfix, file_type)

# 当前特征 + 之前特征 merge 之后的完整训练数据
tr_out_path = output_root_path + 'tr_{}.{}'.format(postfix, file_type)
te_out_path = output_root_path + 'te_{}.{}'.format(postfix, file_type)


ID_NAMES = ['session_id', 'impressions']
TARGET_NAME = 'target'

def feat_extract(df_ori):
    required_cols = ['session_id', 'action_type', 'reference']
    df = df_ori[required_cols]

    actions = ['interaction item image', 'interaction item info', \
            'interaction item deals', 'interaction item rating', \
            'search for item']
    df = df[df.action_type.isin(actions)]
    
    # 过滤掉 reference 中非数值的数据
    print ('filter before', df.shape)
    df['reference'] = df['reference'].astype(str)
    num_index= df['reference'].str.isnumeric()
    df = df[num_index]
    df.rename(columns={'reference': 'impressions'}, inplace=True)
    df['impressions'] = df['impressions'].astype('int') 
    print ('filter after', df.shape)

    df_feat = custom_cate_encoding.gen_hist_feat(df, \
            ID_NAMES, 'action_type', ratio=False)
    print (df_feat.shape)
    print (df_feat.head())

    return df_feat

def output_fea(tr, te):
    # 特征重排，保证输出顺序一致
    # ...

    # 特征文件只保留主键 & 本次新增特征
    #primary_keys = ['session_id', 'impressions']
    #fea_cols = []
    #required_cols =  primary_keys + fea_cols

    # 特征输出
    #tr = tr[required_cols]
    #te = te[required_cols]

    print (tr.head())
    print (te.head())

    loader.save_df(tr, tr_fea_out_path)
    loader.save_df(te, te_fea_out_path)

def filter_acts_after_last_clk(df):
    tr_sample = loader.load_df('../feature/tr_s0_0.ftr')
    te_sample = loader.load_df('../feature/te_s0_0.ftr')
    df_sample = pd.concat([tr_sample, te_sample])

    df_sample = df_sample[['session_id','step']].drop_duplicates()
    df = df.merge(df_sample, on='session_id', how='left')
    print(df.head(10))
    df = df[df.step_x < df.step_y]
    return df

# 生成特征
def gen_fea(base_tr_path=None, base_te_path=None):

    tr = loader.load_df('../input/train.ftr')
    te = loader.load_df('../input/test.ftr')
    df_base = pd.concat([tr, te])

    #df_base = filter_acts_after_last_clk(df_base)

    df_feat = feat_extract(df_base)
    loader.save_df(df_feat, '../feature/df_feat.ftr')

    tr_sample = loader.load_df('../feature/tr_s0_0.ftr')
    te_sample = loader.load_df('../feature/te_s0_0.ftr')

    tr = tr_sample[ID_NAMES].merge(df_feat, on=ID_NAMES, how='left')
    te = te_sample[ID_NAMES].merge(df_feat, on=ID_NAMES, how='left')

    print (tr.shape, te.shape)
    print (tr.head())
    print (te.head())
    print (tr.columns)

    #tr = df_base[pd.notnull(df_base['target'])].reset_index(drop=True)
    #te = df_base[pd.isnull(df_base['target'])].reset_index(drop=True)

    output_fea(tr, te)

# merge 已有特征
def merge_fea(tr_list, te_list):
    tr = loader.merge_fea(tr_list, primary_keys=ID_NAMES)
    te = loader.merge_fea(te_list, primary_keys=ID_NAMES)

    print (tr.head())
    print (te.head())

    loader.save_df(tr, tr_out_path)
    loader.save_df(te, te_out_path)


if __name__ == "__main__":

    print('start time: %s' % datetime.now())
    root_path = '../feature/'
    base_tr_path = root_path + 'tr_s0_0.ftr'
    base_te_path = root_path + 'te_s0_0.ftr'

    gen_fea()

    # merge fea
    prefix = 's0'
    fea_list = [1]

    tr_list = [base_tr_path] + \
            [root_path + 'tr_fea_{}_{}.ftr'.format(prefix, i) for i in fea_list]
    te_list = [base_te_path] + \
            [root_path + 'te_fea_{}_{}.ftr'.format(prefix, i) for i in fea_list]

    #merge_fea(tr_list, te_list)

    print('all completed: %s' % datetime.now())

