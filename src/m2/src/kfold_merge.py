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

# 设置随机种子
SEED = 2018
np.random.seed (SEED)

TARGET_NAME = 'target'
FOLD_NUM = 5

def merge_cv(file_path, sub_name, fold_num):
    file_list = os.listdir(file_path)

    paths = ['{}_cv_{}.csv'.format(sub_name, i) for i in range(1, fold_num + 1)]
    print (paths)

    dfs = []
    for path in paths:
        assert path in file_list, '{} not exist'.format(path)
        path = '{}/{}'.format(file_path, path)
        dfs.append(loader.load_df(path))

    df = pd.concat(dfs)
    print (df.head())
    print (df.describe())
    out_path = '{}/{}_cv.csv'.format(file_path, sub_name)
    df.to_csv(out_path, float_format = '%.4f', index=False)

def merge_sub(file_path, sub_name, fold_num):
    file_list = os.listdir(file_path)

    paths = ['{}_{}.csv'.format(sub_name, i) for i in range(1, fold_num + 1)]
    print (paths)

    df = pd.DataFrame()
    for i, path in enumerate(paths):
        assert path in file_list, '{} not exist'.format(path)
        path = '{}/{}'.format(file_path, path)
        if i == 0:
            df = loader.load_df(path)
        else:
            df[TARGET_NAME] += loader.load_df(path)[TARGET_NAME]

    df[TARGET_NAME] /= fold_num
    print (df.head())
    print (df.describe())
    out_path = '{}/{}.csv'.format(file_path, sub_name)
    df.to_csv(out_path, float_format = '%.4f', index=False)


if __name__ == '__main__':

    sub_file_path = sys.argv[1]
    sub_name = sys.argv[2]

    merge_cv(sub_file_path, sub_name, FOLD_NUM)
    merge_sub(sub_file_path, sub_name, FOLD_NUM)


