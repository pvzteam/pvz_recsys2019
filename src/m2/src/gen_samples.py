#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 基础模块
import os
import sys
import time
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

def gen_tr_feat():
    df = loader.load_df('../input/sample_train.ftr')
    df['reference'] = df['reference'].astype('int')
    df['target'] = (df['reference'] == df['impressions']).astype(int)
    df.drop(['reference','action_type'],axis=1,inplace=True)
    df_session = df[['session_id','step']].drop_duplicates(subset='session_id',keep='last').reset_index(drop=True)
    df = df_session.merge(df, on=['session_id','step'], how='left').reset_index(drop=True)
    loader.save_df(df,'../input/tr.ftr')

def get_te_feat():
    df = loader.load_df('../input/sample_test.ftr')
    df = df[pd.isnull(df['reference'])].reset_index(drop=True)
    print(df.shape)
    df.drop(['reference','action_type'],axis=1,inplace=True)
    loader.save_df(df,'../input/te.ftr')

def gen_semi_feat():
    df = loader.load_df('../input/sample_test.ftr')
    print (df.shape)
    df_session = df[['session_id','step']].drop_duplicates(subset='session_id',keep='last').reset_index(drop=True)
    df = df_session.merge(df, on=['session_id','step'], how='left').reset_index(drop=True)
    print (df.shape)
    df = df[pd.notnull(df['reference'])].reset_index(drop=True)
    print (df.shape)
    df['reference'] = df['reference'].astype('int')
    df['target'] = (df['reference'] == df['impressions']).astype(int)

    df.drop(['reference','action_type'],axis=1,inplace=True)
    loader.save_df(df,'../input/semi.ftr')

def gen_tr_click():
    df = loader.load_df('../input/sample_train.ftr')
    df = df[['session_id','reference']].drop_duplicates(subset='session_id',keep='last').reset_index(drop=True)
    print(df.shape)
    loader.save_df(df,'../input/tr_click.ftr')


if __name__ == '__main__':
    gen_tr_feat()
    get_te_feat()
    gen_tr_click()
    gen_semi_feat()



