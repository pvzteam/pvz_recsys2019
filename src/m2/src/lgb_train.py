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
from lgb_learner import lgbLearner

# 设置随机种子
SEED = 2018
np.random.seed (SEED)

FEA_NUM = sys.argv[1]

fold_num = 5
out_name  = 'lgb_s0_m2_{}'.format(FEA_NUM)
root_path = '../model/' + out_name + '/'

ID_NAMES = ['session_id', 'impressions']
TARGET_NAME = 'target'

if not os.path.exists(root_path):
    os.mkdir(root_path)
    print ('create dir succ {}'.format(root_path))

def get_feas(data):

    cols = data.columns.tolist()
    del_cols = ['target', 'cv', 'session_id', 'user_id', \
            'impressions', 'r1_item']

    if int(FEA_NUM) in [107]:
        # fea 87 imp.delta >= -0.005
        del_cols += ['search for item', 'session_id_by_impr_rank_min', 'impressions_by_hist_clickout item_std', 'impr_rank_1_price', 'impressions_by_impr_rank_min', 'session_hist_filter selection', 'session_id_by_star_std', 'prices_div_impressions_by_prices_max', 'impressions_by_hist_interaction item info_std', 'impressions_by_step_max', 'session_id_by_star_max', 'prices_sub_impressions_by_prices_max', 'prices_sub_impressions_by_prices_min', 'interaction item rating', 'impressions_by_hist_interaction item image_median', 'session_hist_search for poi', 'interactionitem deals', 'session_id_by_impressions_target_max', 'city', 'session_id_by_impressions_target_min', 'interaction item info', 'session_hist_interaction item deals', 'impressions_by_hist_interaction item deals_sum', 'session_hist_change of sort order', 'impressions_by_hist_clickout item_median', 'impressions_by_impr_rank_max', 'impressions_by_hist_interaction item info_median', 'impressions_by_hist_interaction item deals_std', 'impressions_by_hist_interaction item deals_median', 'impressions_by_hist_search for item_median', 'impressions_by_step_min', 'hour', 'timestamp']

    cols = [val for val in cols if val not in del_cols]
    print ('del_cols', del_cols)

    return cols

def lgb_train(train_data, test_data, fea_col_names, seed=SEED, cv_index=0):
    params = {
        "objective":        "binary",
        "boosting_type":    "gbdt",
        #"metric":           ['binary_logloss'],
        "metric":           ['auc'],
        "boost_from_average": False,
        "learning_rate":    0.1,
        "num_leaves":       32,
        "max_depth":        -1,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq":     2,
        "lambda_l1":        0,
        "lambda_l2":        0,
        "seed":             seed,
        'min_child_weight':  0.005,
        'min_data_in_leaf':  50,
        'max_bin':           255,
        "num_threads":       16,
        "verbose":          -1,
        "early_stopping_round": 50
    }

    params['learning_rate'] = 0.02
    params['early_stopping_round'] = 100

    if int(FEA_NUM) in [38, 87]:
        params['learning_rate'] = 0.1
        print ('reset learning_rate to 0.1')

    num_trees = 50000
    print ('training params:', num_trees, params)

    lgb_learner = lgbLearner(train_data, test_data, \
            fea_col_names, ID_NAMES, TARGET_NAME, \
            params, num_trees, fold_num, out_name, \
            metric_names=['auc', 'logloss'], \
            model_postfix='')

    predicted_folds = [1,2,3,4,5]

    lgb_learner.multi_fold_train(lgb_learner.train_data, \
            predicted_folds=predicted_folds, need_predict_test=True)

    #lgb_learner.multi_fold_predict(lgb_learner.train_data, \
    #        predicted_folds=predicted_folds, need_predict_test=True)

if __name__ == '__main__':

    ##################  params ####################
    print("Load the training, test and store data using pandas")
    ts = time.time()
    postfix = 's0_{}'.format(FEA_NUM)
    file_type = 'ftr'

    train_path = '../../../feat/' + 'm2_tr_{}.{}'.format(postfix, file_type)
    test_path  = '../../../feat/' + 'm2_te_{}.{}'.format(postfix, file_type)

    print ('train path', train_path)
    train_data = loader.load_df(train_path)
    test_data = loader.load_df(test_path)

    print (train_data.columns)
    print (train_data.shape, test_data.shape)

    fea_col_names = get_feas(train_data)
    print (len(fea_col_names), fea_col_names)

    required_cols = ID_NAMES + ['cv', 'target']
    drop_cols = [col for col in train_data.columns \
            if col not in fea_col_names and col not in required_cols]

    train_data = train_data.drop(drop_cols, axis=1)
    test_data = test_data.drop(drop_cols, axis=1)

    lgb_train(train_data, test_data, fea_col_names)
    print('all completed: %s, cost %s' % (datetime.now(), time.time() - ts))




