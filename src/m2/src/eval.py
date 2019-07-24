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

from sklearn.metrics import roc_auc_score

def calc_mrr(df):
    df = df[df.label == 1]
    df['rank'] += 1
    df['r_rank'] = 1.0 / df['rank']
    mrr = df['r_rank'].sum() * 1.0 / df.shape[0]
    return mrr


if __name__ == "__main__":

    print('start time: %s' % datetime.now())
    in_path = sys.argv[1]
    df = loader.load_df(in_path)

    auc = roc_auc_score(df['label'], df['target'])
    mrr = calc_mrr(df)

    print ('{}, {}'.format(round(auc, 5), round(mrr, 5)))

    print('all completed: %s' % datetime.now())

