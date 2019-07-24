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

import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

def gen_hist_feat(df_feat, id_names, cate_name, ratio=False):
    df_feat['num'] = 1
    df_cnt = df_feat.groupby(id_names)['num'].agg('sum').reset_index()
    df_feat = df_feat.groupby(id_names + [cate_name]) \
            ['num'].agg('sum').reset_index()
    df_feat = df_feat.set_index(id_names + [cate_name]) \
            [['num']].unstack(level=-1).fillna(0)
    df_feat.columns = df_feat.columns.get_level_values(1)
    df_feat = df_feat.reset_index()

    if ratio:
        df_feat = df_feat.merge(df_cnt,on=id_names, how='left')
        for c in df_feat.columns:
            if c in id_names or c == 'num':
                continue
            df_feat[c] = df_feat[c] / df_feat['num']
        df_feat.drop('num',axis=1,inplace=True)
    return df_feat

def target_encode(df_tr, df_te, cols, target):
    """https://github.com/scikit-learn-contrib/categorical-encoding/blob/master/category_encoders/target_encoder.py"""
    te = ce.TargetEncoder(cols=cols, return_df=True, min_samples_leaf=5) \
            .fit(df_tr[cols], df_tr[target])
    feat_tr = te.transform(df_tr[cols], df_tr[target])
    feat_te = te.transform(df_te[cols])
    return feat_tr, feat_te

def cate_label_encoding(df_ori, col_names, target_name, alpha=10):
    df = df_ori.copy()
    df_obj = df[['cv'] + col_names].drop_duplicates()

    target = target_name
    dfs = []
    for fold in range(0, 6):
        mask = (df.cv != fold) & (df.cv != 0)
        oof_df = df.loc[mask]
        prior = oof_df[target].mean()

        agg_df = oof_df.groupby(col_names)[target].agg(["sum", "count"])
        agg_df[target] = (agg_df["sum"] + alpha * prior) / (agg_df["count"] + alpha)
        agg_df.drop(['sum', 'count'], axis=1, inplace=True)

        cur_df = df_obj[df_obj.cv == fold]
        cur_df = cur_df.merge(agg_df, left_on=col_names, right_index=True, how='left')
        dfs.append(cur_df)
        print (fold, cur_df.shape)

    df_obj = df_obj.merge(pd.concat(dfs), on=['cv'] + col_names, how='left')
    prefix = '_'.join(col_names)
    df_obj.rename(columns={target : '{}_target'.format(prefix)}, inplace=True)
    #df_obj.to_csv('tmp.csv', index=False)
    print (df_obj.shape)
    print (df_obj.head(20))
    return df_obj
