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

FEA_NUM = 8

input_root_path = '../input/'
output_root_path = '../feature/'

tr_base_path = input_root_path + 'train.ftr'
te_base_path = input_root_path + 'test.ftr'

cv_id_path = input_root_path + 'cv_id.csv.0329'

postfix = 's0_{}'.format(FEA_NUM)
file_type = 'ftr'

# 当前特征
tr_fea_out_path = output_root_path + 'tr_fea_{}.{}'.format(postfix, file_type)
te_fea_out_path = output_root_path + 'te_fea_{}.{}'.format(postfix, file_type)

# 当前特征 + 之前特征 merge 之后的完整训练数据
tr_out_path = output_root_path + 'tr_{}.{}'.format(postfix, file_type)
te_out_path = output_root_path + 'te_{}.{}'.format(postfix, file_type)


ID_NAMES = ['session_id', 'impressions']
TARGET_NAME = 'target'

def feat_extract(df):
    df['impr_rank'] = df.groupby(['session_id']).cumcount().values
    df_pos1 = df[df.impr_rank == 0]
    df_pos1 = df_pos1[['session_id', 'prices']]
    df_pos1.columns = ['session_id', 'impr_rank_1_price']

    df_feat = df[ID_NAMES + ['prices']].merge(df_pos1, on='session_id', how='left')
    df_feat['price_sub_impr_rank_1_price'] = df_feat['prices'] - \
            df_feat['impr_rank_1_price']
    df_feat['price_div_impr_rank_1_price'] = df_feat['prices'] / \
            df_feat['impr_rank_1_price']
    del df_feat['prices']

    print (df_feat.shape)
    print (df_feat.head())
    print (df_feat.columns.tolist())

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

# 生成特征
def gen_fea():

    #tr = loader.load_df('../input/train.ftr')
    #te = loader.load_df('../input/test.ftr')

    tr = loader.load_df('../input/tr.ftr')
    te = loader.load_df('../input/te.ftr')

    df_base = pd.concat([tr, te])
    df_feat = feat_extract(df_base)

    tr_sample = loader.load_df('../feature/tr_s0_0.ftr')
    te_sample = loader.load_df('../feature/te_s0_0.ftr')

    merge_keys = ['session_id', 'impressions']
    tr = tr_sample[ID_NAMES].merge(df_feat, on=merge_keys, how='left')
    te = te_sample[ID_NAMES].merge(df_feat, on=merge_keys, how='left')

    print (tr.shape, te.shape)
    print (tr.head())
    print (te.head())
    print (tr.columns)

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
    base_tr_path = root_path + 'tr_s0_6.ftr'
    base_te_path = root_path + 'te_s0_6.ftr'

    gen_fea()

    # merge fea
    prefix = 's0'
    fea_list = [FEA_NUM]

    tr_list = [base_tr_path] + \
            [root_path + 'tr_fea_{}_{}.ftr'.format(prefix, i) for i in fea_list]
    te_list = [base_te_path] + \
            [root_path + 'te_fea_{}_{}.ftr'.format(prefix, i) for i in fea_list]

    #merge_fea(tr_list, te_list)

    print('all completed: %s' % datetime.now())

