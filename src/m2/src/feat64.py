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
from sklearn.feature_extraction.text import CountVectorizer

# 自定义工具包
sys.path.append('../tools/')
import loader
import cate_encoding
import custom_cate_encoding

# 设置随机种子
SEED = 2018
np.random.seed (SEED)

FEA_NUM = 64

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
    tr = loader.load_df('../input/tr.ftr')
    te = loader.load_df('../input/te.ftr')
    df_sample = pd.concat([tr, te])

    df['star'] = df.properties.apply(lambda s : \
            ''.join([v for v in s.split('|') if 'Star' in v and 'Stars' not in v]))
    df['star'] = df['star'].apply(lambda s : s.split(' ')[0])
    df['star'].replace('', 0, inplace=True)
    print (df['star'].value_counts())
    df_star = df[['item_id', 'star']].drop_duplicates(subset=['item_id'])
    df_star.columns = ['impressions', 'star']
    df_star['star'] = df_star['star'].astype('int')
  
    df_feat = df_sample[ID_NAMES + ['prices']].drop_duplicates(subset=ID_NAMES) \
            .merge(df_star, on='impressions', how='left')

    df_feat = cate_encoding.cate_num_stat(df_feat, df_feat, \
            ['session_id'], 'star', ['max', 'median', 'std'])

    df_feat['star_sub_session_max'] = \
            df_feat['star'] - df_feat['session_id_by_star_max']
    df_feat['star_sub_session_median'] = \
            df_feat['star'] - df_feat['session_id_by_star_median']

    df_feat['star_div_prices'] = df_feat['star'] * 100 / df_feat['prices']
    print (df_feat['star_div_prices'].describe())

    df_feat = cate_encoding.cate_num_rank(df_feat, \
            ['session_id'], 'star_div_prices', ascending=False)
    del df_feat['prices']

    print ('df_feat info')
    print (df_feat.shape)
    print (df_feat.head())
    print (df_feat.columns.tolist())

    return df_feat

def output_fea(tr, te):
    print (tr.head())
    print (te.head())

    loader.save_df(tr, tr_fea_out_path)
    loader.save_df(te, te_fea_out_path)

def add_meta_fea(df):
    pass

# 生成特征
def gen_fea(base_tr_path=None, base_te_path=None):

    #tr = loader.load_df('../input/train.ftr')
    #te = loader.load_df('../input/test.ftr')

    #tr = loader.load_df('../input/tr.ftr')
    #te = loader.load_df('../input/te.ftr')

    #tr = loader.load_df('../feature/tr_s0_0.ftr')
    #te = loader.load_df('../feature/te_s0_0.ftr')

    #tr = loader.load_df('../feature/tr_fea_s0_1.ftr')
    #te = loader.load_df('../feature/te_fea_s0_1.ftr')

    #tr = tr.head(1000)
    #te = te.head(1000)

    #df_base = pd.concat([tr, te])
    df_base = loader.load_df('../input/item_metadata.ftr')
    #df_base = df_base.head(1000)
    df_feat = feat_extract(df_base)

    tr_sample = loader.load_df('../feature/tr_s0_0.ftr')
    te_sample = loader.load_df('../feature/te_s0_0.ftr')

    merge_keys = ['session_id', 'impressions']
    #merge_keys = ['session_id']
    #merge_keys = ['impressions']

    tr = tr_sample[ID_NAMES].merge(df_feat, on=merge_keys, how='left')
    te = te_sample[ID_NAMES].merge(df_feat, on=merge_keys, how='left')

    float_cols = [c for c in tr.columns if tr[c].dtype == 'float']
    tr[float_cols] = tr[float_cols].astype('float32')
    te[float_cols] = te[float_cols].astype('float32')

    print (tr.shape, te.shape)
    print (tr.head())
    print (te.head())
    print (tr.columns)

    output_fea(tr, te)

# merge 已有特征
def merge_fea(tr_list, te_list):
    tr = loader.merge_fea(tr_list, primary_keys=ID_NAMES)
    te = loader.merge_fea(te_list, primary_keys=ID_NAMES)

    tr['impressions'] = tr['impressions'].astype('int')
    te['impressions'] = te['impressions'].astype('int')

    print (tr.head())
    print (te.head())

    print (tr[ID_NAMES].head())

    loader.save_df(tr, tr_out_path)
    loader.save_df(te, te_out_path)


if __name__ == "__main__":

    print('start time: %s' % datetime.now())
    root_path = '../feature/'
    base_tr_path = root_path + 'tr_s0_35.ftr'
    base_te_path = root_path + 'te_s0_35.ftr'

    gen_fea()

    # merge fea
    prefix = 's0'
    #fea_list = [3,6,8,14,15,FEA_NUM]
    fea_list = [FEA_NUM]

    tr_list = [base_tr_path] + \
            [root_path + 'tr_fea_{}_{}.ftr'.format(prefix, i) for i in fea_list]
    te_list = [base_te_path] + \
            [root_path + 'te_fea_{}_{}.ftr'.format(prefix, i) for i in fea_list]

    #merge_fea(tr_list, te_list)

    print('all completed: %s' % datetime.now())

