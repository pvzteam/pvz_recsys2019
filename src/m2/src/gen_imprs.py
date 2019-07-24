#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 基础模块
import os
import sys
from datetime import datetime

# 数据处理
import numpy as np
import pandas as pd

# 自定义工具包
sys.path.append('../tools/')
import loader

def string_to_array(s):
    """Convert pipe separated string to array."""

    if isinstance(s, str):
        out = s.split("|")
    elif math.isnan(s):
        out = []
    else:
        raise ValueError("Value must be either string of nan")
    return out


def explode(df_in, col_expls):
    """Explode column col_expl of array type into multiple rows."""

    df = df_in.copy()
    for col_expl in col_expls:
        df.loc[:, col_expl] = df[col_expl].apply(string_to_array)
    
    base_cols = list(set(df.columns) - set(col_expls))
    df_out = pd.DataFrame(
        {col: np.repeat(df[col].values,
                        df[col_expls[0]].str.len())
         for col in base_cols}
    )

    for col_expl in col_expls:
        df_out.loc[:, col_expl] = np.concatenate(df[col_expl].values)
        df_out.loc[:, col_expl] = df_out[col_expl].apply(int)
    return df_out

def gen_sample(ori, des):
    df = loader.load_df(ori)
    print(df.shape)
    df = df[df.action_type=='clickout item']
    print(df.shape)
    df_out = explode(df, ['impressions', 'prices'])
    print(df_out.shape)
    loader.save_df(df_out, des)

if __name__ == '__main__':
    print ('gen train')
    gen_sample('../input/train.ftr', '../input/sample_train.ftr')
    print ('gen test')
    gen_sample('../input/test.ftr', '../input/sample_test.ftr')


