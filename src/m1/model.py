# -*- coding: utf-8 -*-

import gc
import numpy as np 
import pandas as pd 

import lightgbm as lgb

from data import *
from feat import *
from resource import *
from utils import (load_dataframe, convert_dtype, CrossValidation, merge_all)


def rank_feat_inside_session(df, cols):
    for col in cols:
        col_n = '%s_rank' % col

        df[col_n] = df.groupby(['session_id'])[col].rank(method='first').astype('float32')
        
    return df


def rank_similarity_inside_session(df):
    cols_rank = [
        'item_id_interaction_last_similarity_wv_impression', 
        'item_id_impression_prev_item_meta_cos', 
        'item_id_impression_prev_co_appearence_impression_count', 
        'item_id_impression_first_co_appearence_interaction_count', 
        'item_id_impression_first_co_appearence_impression_count', 
        'item_id_interaction_last_co_appearence_impression_count', 
        'item_id_interaction_last_co_appearence_interaction_count', 
    ]
    
    return rank_feat_inside_session(df, cols_rank)


f_m2_top30 = TrainTestResource(FeatResource, 'm2_%s_top30_fea', 
                               fix=['tr', 'te'], fmt='ftr')
f_m3_top30 = TrainTestResource(FeatResource, 'm3_%s_feat_top30', 
                               fix=['tr', 'te'], fmt='ftr')

m_20190622 = TrainTestResource(ModelResource, '%s_m1_20190622', 
                               fix=['tr', 'te'], fmt='csv')
m_20190624 = TrainTestResource(ModelResource, '%s_m1_20190624', 
                               fix=['tr', 'te'], fmt='csv')
m_20190626 = TrainTestResource(ModelResource, '%s_m1_20190626', 
                               fix=['tr', 'te'], fmt='csv')


@register(out=m_20190622, inp=[t_tr_te_classify, f_top100, f_si_sim])
def train_predict_lgb_20190622_2():
    from feat_names import names_lgb_20190622_2 as feats

    def load_data(tt):
        df = merge_all([
            t_tr_te_classify[tt].load().rename(columns={'item_id': 'impressions'}),
            f_top100[tt].load(),
            f_si_sim[tt].load().rename(columns={'item_id': 'impressions'}),
        ], on=['session_id', 'impressions'], how='left')

        df = rank_similarity_inside_session(df)
        return df

    train = load_data('train')
    cv = CrossValidation()

    model = lgb.LGBMClassifier(n_estimators=50000, objective="binary", metric='binary_logloss', 
                               num_leaves=31, min_child_samples=100, learning_rate=0.1, 
                               bagging_fraction=0.7, feature_fraction=0.7, bagging_frequency=5, 
                               seed=1, feature_fraction_seed=1, use_best_model=True, n_jobs=16)

    df_train = train[['session_id', 'impressions']]
    df_train['target'] = cv.validate(model, feats, train, train['target'], early_stopping_rounds=100, verbose=100)
    print('Validation Score:', np.mean(cv.scores))

    del train 
    gc.collect()

    test = load_data('test')
    df_test = test[['session_id', 'impressions']]
    df_test['target'] = cv.predict_proba(test)

    df_train.to_csv(m_20190622.train.path, index=False, float_format='%.4f')
    df_test.to_csv(m_20190622.test.path, index=False, float_format='%.4f')
    
    
@register(out=m_20190624, inp=[t_tr_te_classify, f_top30, f_si_sim, 
                               f_m2_top30, f_m3_top30])
def train_predict_lgb_20190624_1():
    from feat_names import names_lgb_20190624_1 as feats

    def load_data(tt):
        df = merge_all([
            t_tr_te_classify[tt].load().rename(columns={'item_id': 'impressions'}),
            f_m2_top30[tt].load(),
            f_m3_top30[tt].load(),
            f_top30[tt].load(),
            f_si_sim[tt].load().rename(columns={'item_id': 'impressions'}),
        ], on=['session_id', 'impressions'], how='left')

        df = rank_similarity_inside_session(df)
        return df

    train = load_data('train')
    cv = CrossValidation()

    model = lgb.LGBMClassifier(n_estimators=50000, objective="binary", metric='binary_logloss', 
                               num_leaves=31, min_child_samples=100, learning_rate=0.1, 
                               bagging_fraction=0.7, feature_fraction=0.7, bagging_frequency=5, 
                               seed=1, use_best_model=True, n_jobs=16)

    df_train = train[['session_id', 'impressions']]
    df_train['target'] = cv.validate(model, feats, train, train['target'], early_stopping_rounds=100, verbose=100)
    print('Validation Score:', np.mean(cv.scores))

    del train 
    gc.collect()

    test = load_data('test')
    df_test = test[['session_id', 'impressions']]
    df_test['target'] = cv.predict_proba(test)
    
    df_train.to_csv(m_20190624.train.path, index=False, float_format='%.4f')
    df_test.to_csv(m_20190624.test.path, index=False, float_format='%.4f') 


@register(out=m_20190626, inp=[t_tr_te_classify, f_top30, f_si_sim, 
                               f_si_cmp, f_si_win,
                               f_m2_top30, f_m3_top30])
def train_predict_lgb_20190626_2():
    from feat_names import names_lgb_20190626_2 as feats

    def load_data(tt):
        df = merge_all([
            t_tr_te_classify[tt].load().rename(columns={'item_id': 'impressions'}),
            f_m2_top30[tt].load(),
            f_m3_top30[tt].load(),
            f_top30[tt].load(),
            f_si_sim[tt].load().rename(columns={'item_id': 'impressions'}),
            f_si_cmp[tt].load().rename(columns={'item_id': 'impressions'}),
            f_si_win[tt].load().rename(columns={'item_id': 'impressions'}),
        ], on=['session_id', 'impressions'], how='left')

        df = rank_similarity_inside_session(df)
        
        cols_win = [
            'item_id_impression_prev_item_win_ratio',
            'item_id_impression_first_item_win_ratio',
            'item_id_interaction_last_item_win_ratio',
            'item_id_interaction_most_item_win_ratio',
        ]
        
        df = rank_feat_inside_session(df, cols_win)
        return df

    train = load_data('train')
    cv = CrossValidation()

    model = lgb.LGBMClassifier(n_estimators=50000, objective="binary", metric='binary_logloss', 
                               num_leaves=31, min_child_samples=100, learning_rate=0.1, 
                               bagging_fraction=0.7, feature_fraction=0.7, bagging_frequency=5, 
                               seed=1, feature_fraction_seed=1, use_best_model=True, n_jobs=16)

    df_train = train[['session_id', 'impressions']]
    df_train['target'] = cv.validate(model, feats, train, train['target'], early_stopping_rounds=100, verbose=100)
    print('Validation Score:', np.mean(cv.scores))

    del train 
    gc.collect()

    test = load_data('test')
    df_test = test[['session_id', 'impressions']]
    df_test['target'] = cv.predict_proba(test)
    
    df_train.to_csv(m_20190626.train.path, index=False, float_format='%.4f')
    df_test.to_csv(m_20190626.test.path, index=False, float_format='%.4f')
    
    
