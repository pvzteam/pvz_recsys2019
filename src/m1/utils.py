# -*- coding: utf-8 -*-

import os
import gc
import sys
import time
import datetime
import functools

import feather
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.externals import joblib
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score


def mock_name(name, prefix='m1'):
    return prefix + '_' + name


def load_dataframe(path, fmt=None, columns=None):
    if not fmt:
        if path.endswith('tsv'):
            fmt = 'tsv'
        elif path.endswith('feather') or path.endswith('ftr'):
            fmt = 'feather'
        else:
            fmt = 'csv'

    if fmt == 'tsv':
        return pd.read_csv(path, '\t', usecols=columns)
    elif fmt in ('ftr', 'feather'):
        import feather
        return feather.read_dataframe(path, columns=columns)
    else:
        return pd.read_csv(path, usecols=columns)
    

def save_dataframe(df, path, fmt=None, *args, **kwargs):
    if not fmt:
        if path.endswith('tsv'):
            fmt = 'tsv'
        elif path.endswith('feather') or path.endswith('ftr'):
            fmt = 'feather'
        else:
            fmt = 'csv'

    if kwargs.get('columns'):
        df = df[kwargs.get('columns')]

    if fmt in ('ftr', 'feather'):
        df.to_feather(path)
    elif fmt == 'tsv':
        df.to_csv(path, '\t', *args, **kwargs)
    else:
        df.to_csv(path, *args, **kwargs)


def merge_all(df_lst, *args, **kwargs):
    df = None
    for dd in df_lst:
        if df is None:
            df = dd
        else:
            df = pd.merge(df, dd, *args, **kwargs)
            
    return df


def astype(df, columns, dtype_t, cols_skip=['session_id', 'item_id']):
    for col in columns:
        if col in cols_skip:
            continue

        df[col] = df[col].astype(dtype_t)

        
def astype_int8(df, columns,  cols_skip=['session_id', 'item_id'], fillna=0):
    for col in columns:
        if col in cols_skip:
            continue

        df[col] = df[col].fillna(fillna).astype('int8')


def convert_dtype(df, dtype_dict=None):
    if not dtype_dict:
        dtype_dict = {'float64': 'float32', 'int64': 'int32'}
    
    for col in df.columns:
        dtype_s = str(df[col].dtype)
        if dtype_s not in dtype_dict:
            continue
        
        dtype_t = dtype_dict[dtype_s]
        if isinstance(dtype_t, list) or isinstance(dtype_t, tuple):
            dtype_t_lst = list(dtype_t)
        else:
            dtype_t_lst = [dtype_t]

        for dtype_t in dtype_t_lst:
            try:
                df[col] = df[col].astype(dtype_t)
                break
            except:
                pass
        else:
            print('Cannot convert dtype for column %s' % col)

    gc.collect()
    return df


def string_to_array(s):
    """Convert pipe separated string to array."""

    if isinstance(s, str):
        out = s.split("|")
    elif math.isnan(s):
        out = []
    else:
        raise ValueError("Value must be either string of nan")
    return out


def explode(df_in, col_expl):
    """Explode column col_expl of array type into multiple rows."""

    df = df_in.copy()
    df.loc[:, col_expl] = df[col_expl].apply(string_to_array)

    df_out = pd.DataFrame(
        {col: np.repeat(df[col].values,
                        df[col_expl].str.len())
         for col in df.columns.drop(col_expl)}
    )

    df_out.loc[:, col_expl] = np.concatenate(df[col_expl].values)
    df_out.loc[:, col_expl] = df_out[col_expl].apply(int)

    return df_out


class CrossValidation(object):
    def __init__(self, **kwargs):
        self.df_fi = None

        self.features = None
        self.models = []
        self.scores = []

    def predict_proba(self, X, return_all=False):
        y_proba = np.zeros([X.shape[0], len(self.models)])

        for fold_id, model_ in enumerate(self.models):
            y_proba[:, fold_id] = model_.predict_proba(X[self.features])[:,1].reshape(-1)

        if return_all:
            return y_proba
        else:
            return y_proba.mean(axis=1)

    def feature_importance(self, fold_id=None):
        if fold_id is None:
            return self.df_fi.mean(axis=1).sort_values(ascending=False)
        else:
            return self.df_fi['fold_%s' % fold_id].sort_values(ascending=False)

    def validate(self, model, features, X, y, **fit_params):
        y_proba = np.zeros(X.shape[0])

        self.features = features
        self.models = []
        self.scores = []
        self.df_fi = pd.DataFrame(index=self.features)
        for fold_id in range(5):
            print("Fold ", fold_id, ":", datetime.datetime.now())
            
            mock_fold_val = X['fold'] == fold_id
            cv_train_X = X[~mock_fold_val][self.features]
            cv_train_y = y[~mock_fold_val]

            cv_val_X = X[mock_fold_val][self.features]
            cv_val_y = y[mock_fold_val]

            model_ = clone(model)
            model_.fit(cv_train_X, cv_train_y,
                       eval_set=[(cv_val_X, cv_val_y)], **fit_params)

            y_proba[mock_fold_val] = model_.predict_proba(cv_val_X)[:,1].reshape(-1)
            score_ = roc_auc_score(cv_val_y, y_proba[mock_fold_val])
            print('Fold Score:', score_)
            
            del mock_fold_val, cv_train_X, cv_train_y, cv_val_X, cv_val_y
            gc.collect()
            
            self.scores.append(score_)
            self.models.append(model_)

            self.df_fi['fold_' + str(fold_id)] = model_.booster_.feature_importance(importance_type='gain')

        return y_proba
    
    
    