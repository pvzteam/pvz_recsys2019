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

FEA_NUM = 108

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

tr_fea_out_path = '../../../feat/' + 'm2_tr_pairwise_fea_{}.{}'.format(postfix, file_type)
te_fea_out_path = '../../../feat/' + 'm2_te_pairwise_fea_{}.{}'.format(postfix, file_type)

# 当前特征 + 之前特征 merge 之后的完整训练数据
tr_out_path = output_root_path + 'tr_{}.{}'.format(postfix, file_type)
te_out_path = output_root_path + 'te_{}.{}'.format(postfix, file_type)


ID_NAMES = ['session_id', 'impressions']
TARGET_NAME = 'target'

def feat_extract(df):
    df = df.sort_values(by='prices')
    df['price_rank'] = df.groupby(['session_id']).cumcount().values

    df = df[ID_NAMES + ['impr_rank', 'price_rank']].drop_duplicates(subset=ID_NAMES)

    tr = loader.load_df('../../../model/m3_71/tr_pred.csv')
    te = loader.load_df('../../../model/m3_71/te_pred.csv')
    df_pred = pd.concat([tr, te]).drop_duplicates(subset=ID_NAMES)
    df_pred = df_pred[ID_NAMES + ['target']]
    
    df_pred = df_pred.sort_values(by=['target'], ascending=False)
    df_pred['pred_rank'] = df_pred.groupby(['session_id']).cumcount().values
    df_pred = df_pred.sort_values(by=['session_id', 'target'])
    print (df_pred.shape)
    print (df_pred.head(10))

    pred_top1 = df_pred[df_pred['pred_rank'] == 0] \
            .drop_duplicates(subset='session_id', keep='first')
    pred_top1 = pred_top1[['session_id', 'target']]
    pred_top1.columns = ['session_id', 'top1_pred']

    pred_top2 = df_pred[df_pred['pred_rank'] < 2]
    pred_top2['top2_pred_avg'] = pred_top2.groupby('session_id')['target'].transform('mean')
    pred_top2['top2_pred_std'] = pred_top2.groupby('session_id')['target'].transform('std')
    pred_top2 = pred_top2[['session_id', 'top2_pred_avg', \
            'top2_pred_std']].drop_duplicates(subset=['session_id'])

    pred_top3 = df_pred[df_pred['pred_rank'] < 3]
    pred_top3['top3_pred_avg'] = pred_top3.groupby('session_id')['target'].transform('mean')
    pred_top3['top3_pred_std'] = pred_top3.groupby('session_id')['target'].transform('std')
    pred_top3 = pred_top3[['session_id', 'top3_pred_avg', \
            'top3_pred_std']].drop_duplicates(subset=['session_id'])

    pred_top5 = df_pred[df_pred['pred_rank'] < 5]
    pred_top5['top5_pred_avg'] = pred_top5.groupby('session_id')['target'].transform('mean')
    pred_top5['top5_pred_std'] = pred_top5.groupby('session_id')['target'].transform('std')
    pred_top5 = pred_top5[['session_id', 'top5_pred_avg', \
            'top5_pred_std']].drop_duplicates(subset=['session_id'])

    df_pred.rename(columns={'target': 'pred'}, inplace=True)
    df = df.merge(df_pred, on=ID_NAMES, how='left')
    df = df.merge(pred_top1, on=['session_id'], how='left')
    df = df.merge(pred_top2, on=['session_id'], how='left')
    df = df.merge(pred_top3, on=['session_id'], how='left')
    df = df.merge(pred_top5, on=['session_id'], how='left')
    
    df['pred_sub_top1'] = df['pred'] - df['top1_pred']
    df['pred_sub_top2_avg'] = df['pred'] - df['top2_pred_avg']
    df['pred_sub_top3_avg'] = df['pred'] - df['top3_pred_avg']
    df['pred_sub_top5_avg'] = df['pred'] - df['top5_pred_avg']

    # impr rank
    pred_impr_top_1 = df[df['impr_rank'] == 0]
    pred_impr_top_1 = pred_impr_top_1[['session_id', 'pred']]
    pred_impr_top_1.columns = ['session_id', 'impr_top1_pred']

    pred_impr_top_2 = df[df['impr_rank'] < 2]
    pred_impr_top_2['impr_top2_pred_avg'] = \
            pred_impr_top_2.groupby('session_id')['pred'].transform('mean')
    pred_impr_top_2['impr_top2_pred_std'] = \
            pred_impr_top_2.groupby('session_id')['pred'].transform('std')
    pred_impr_top_2 = pred_impr_top_2[['session_id', 'impr_top2_pred_avg', 'impr_top2_pred_std']] \
            .drop_duplicates(subset=['session_id'])

    pred_impr_top_3 = df[df['impr_rank'] < 3]
    pred_impr_top_3['impr_top3_pred_avg'] = \
            pred_impr_top_3.groupby('session_id')['pred'].transform('mean')
    pred_impr_top_3['impr_top3_pred_std'] = \
            pred_impr_top_3.groupby('session_id')['pred'].transform('std')
    pred_impr_top_3 = pred_impr_top_3[['session_id', 'impr_top3_pred_avg', 'impr_top3_pred_std']] \
            .drop_duplicates(subset=['session_id'])

    pred_impr_top_5 = df[df['impr_rank'] < 5]
    pred_impr_top_5['impr_top5_pred_avg'] = \
            pred_impr_top_5.groupby('session_id')['pred'].transform('mean')
    pred_impr_top_5['impr_top5_pred_std'] = \
            pred_impr_top_5.groupby('session_id')['pred'].transform('std')
    pred_impr_top_5 = pred_impr_top_5[['session_id', 'impr_top5_pred_avg', 'impr_top5_pred_std']] \
            .drop_duplicates(subset=['session_id'])

    df = df.merge(pred_impr_top_1, on=['session_id'], how='left')
    df = df.merge(pred_impr_top_2, on=['session_id'], how='left')
    df = df.merge(pred_impr_top_3, on=['session_id'], how='left')
    df = df.merge(pred_impr_top_5, on=['session_id'], how='left')
    df['pred_sub_impr_top1'] = df['pred'] - df['impr_top1_pred']
    df['pred_sub_impr_top2_avg'] = df['pred'] - df['impr_top5_pred_avg']
    df['pred_sub_impr_top3_avg'] = df['pred'] - df['impr_top5_pred_avg']
    df['pred_sub_impr_top5_avg'] = df['pred'] - df['impr_top5_pred_avg']

    # price rank
    pred_price_top_1 = df[df['price_rank'] == 0]
    pred_price_top_1 = pred_price_top_1[['session_id', 'pred']]
    pred_price_top_1.columns = ['session_id', 'price_top1_pred']

    pred_price_top_3 = df[df['price_rank'] < 3]
    pred_price_top_3['price_top3_pred_avg'] = \
            pred_price_top_3.groupby('session_id')['pred'].transform('mean')
    pred_price_top_3['price_top3_pred_std'] = \
            pred_price_top_3.groupby('session_id')['pred'].transform('std')
    pred_price_top_3 = pred_price_top_3[['session_id', 'price_top3_pred_avg', 'price_top3_pred_std']] \
            .drop_duplicates(subset=['session_id'])

    pred_price_top_5 = df[df['price_rank'] < 5]
    pred_price_top_5['price_top5_pred_avg'] = \
            pred_price_top_5.groupby('session_id')['pred'].transform('mean')
    pred_price_top_5['price_top5_pred_std'] = \
            pred_price_top_5.groupby('session_id')['pred'].transform('std')
    pred_price_top_5 = pred_price_top_5[['session_id', 'price_top5_pred_avg', 'price_top5_pred_std']] \
            .drop_duplicates(subset=['session_id'])

    df = df.merge(pred_price_top_1, on=['session_id'], how='left')
    df = df.merge(pred_price_top_3, on=['session_id'], how='left')
    df = df.merge(pred_price_top_5, on=['session_id'], how='left')
    df['pred_sub_price_top1'] = df['pred'] - df['price_top1_pred']
    df['pred_sub_price_top3_avg'] = df['pred'] - df['price_top3_pred_avg']
    df['pred_sub_price_top5_avg'] = df['pred'] - df['price_top5_pred_avg']

    del_cols = ['impr_rank', 'price_rank']
    df.drop(del_cols, axis=1, inplace=True)
    df_feat = df.drop_duplicates(subset=ID_NAMES)

    print ('df_feat info')
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

def add_meta_fea(df):
    pass

# 生成特征
def gen_fea(base_tr_path=None, base_te_path=None):

    #tr = loader.load_df('../input/train.ftr')
    #te = loader.load_df('../input/test.ftr')

    #tr = loader.load_df('../input/tr.ftr')
    #te = loader.load_df('../input/te.ftr')

    tr = loader.load_df('../feature/tr_s0_0.ftr')
    te = loader.load_df('../feature/te_s0_0.ftr')

    #tr = loader.load_df('../feature/tr_fea_s0_1.ftr')
    #te = loader.load_df('../feature/te_fea_s0_1.ftr')

    #tr = tr.head(1000)
    #te = te.head(1000)

    df_base = pd.concat([tr, te])
    df_feat = feat_extract(df_base)

    tr_sample = loader.load_df('../feature/tr_s0_0.ftr')
    te_sample = loader.load_df('../feature/te_s0_0.ftr')

    merge_keys = ['session_id', 'impressions']
    #merge_keys = ['session_id']

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
    tr = loader.merge_fea_v2(tr_list, primary_keys=ID_NAMES)
    te = loader.merge_fea_v2(te_list, primary_keys=ID_NAMES)

    tr['impressions'] = tr['impressions'].astype('int')
    te['impressions'] = te['impressions'].astype('int')

    tr_sample = loader.load_df('../feature/tr_s0_0.ftr')
    tr_sample = tr_sample[ID_NAMES + ['cv']]

    tr = tr.merge(tr_sample, on=ID_NAMES, how='left')
    te['cv'] = 0

    print (tr.head())
    print (te.head())

    print (tr[ID_NAMES].head())

    loader.save_df(tr, tr_out_path)
    loader.save_df(te, te_out_path)


if __name__ == "__main__":

    print('start time: %s' % datetime.now())
    root_path = '../feature/'
    base_tr_path = root_path + 'tr_s0_95.ftr'
    base_te_path = root_path + 'te_s0_95.ftr'

    gen_fea()

    # merge fea
    prefix = 's0'
    #fea_list = [3,6,8,14,15,FEA_NUM]
    fea_list = [105,106,FEA_NUM]

    tr_list = [base_tr_path] + \
            [root_path + 'tr_fea_{}_{}.ftr'.format(prefix, i) for i in fea_list]
    te_list = [base_te_path] + \
            [root_path + 'te_fea_{}_{}.ftr'.format(prefix, i) for i in fea_list]

    #merge_fea(tr_list, te_list)

    print('all completed: %s' % datetime.now())

