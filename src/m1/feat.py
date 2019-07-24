# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 

from data import *
from resource import *
from utils import (load_dataframe, explode, merge_all, convert_dtype)
from utils import mock_name as _mm


f_item_meta_gte_50000 = FeatResource(_mm('item_metadata_gte_50000'))
f_item_int_cnt = FeatResource(_mm('item_interaction_cnt'))
f_item_expo_cnt = FeatResource(_mm('item_exposure_count'))

f_sess_basic = TrainTestResource(FeatResource, _mm('session_basic_%s'))
f_sess_int = TrainTestResource(FeatResource, _mm('session_interaction_%s'))
f_sess_imp_eq = TrainTestResource(FeatResource, _mm('session_impressions_eq_%s'))
f_sess_imp = TrainTestResource(FeatResource, _mm('session_imp_%s'))
f_sess_int_price = TrainTestResource(FeatResource, _mm('session_interaction_price_%s'))
f_sess_le = TrainTestResource(FeatResource, _mm('session_label_encode_%s'))

f_si_basic = TrainTestResource(FeatResource, _mm('session_item_basic_%s'))
f_si_first_last = TrainTestResource(FeatResource, _mm('session_item_first_last_%s'))
f_si_int = TrainTestResource(FeatResource, _mm('session_item_interaction_%s'))
f_si_diff_last_int = TrainTestResource(FeatResource, _mm('session_item_diff_last_interaction_index_%s'))
f_si_diff_imp_price = TrainTestResource(FeatResource, _mm('session_item_price_div_imp_price_%s'))

f_top30 = TrainTestResource(FeatResource, _mm('top_30_feats_%s'))
f_top100 = TrainTestResource(FeatResource, _mm('top_100_feats_%s'))

f_si_sim = TrainTestResource(FeatResource, _mm('similarity_pair_%s'))
f_si_cmp = TrainTestResource(FeatResource, _mm('compare_pair_%s'))
f_si_win = TrainTestResource(FeatResource, _mm('win_pair_%s'))


@register(out=f_item_meta_gte_50000, inp=t_item_metadata_sp)
def item_metadata_gte_50000():
    dd = t_item_metadata_sp.load()

    cols_meta = [col for col in dd.columns if col != 'item_id']
    meta_count = dd[cols_meta].sum()
    meta_keep = list(meta_count[meta_count >= 50000].index)

    dd = dd[['item_id'] + meta_keep]
    dd['item_id'] = dd['item_id'].astype(int)
    for col in meta_keep:
        dd[col] = dd[col].astype('float32')

    f_item_meta_gte_50000.save(dd)


@register(out=f_item_int_cnt, inp=i_tr_te)
def item_interaction_cnt():
    cols_keep = ['user_id', 'session_id', 'reference', 'action_type']
    df_train = i_tr_te.train.load(columns=cols_keep)
    df_test = i_tr_te.test.load(columns=cols_keep)
    df = pd.concat([df_train, df_test])

    df = df[df['action_type'] != 'clickout item']
    df = df[(~df['reference'].isnull()) & 
            (df['reference'].str.isdigit())]
    df = df.rename(columns={'reference': 'item_id'})
    df['item_id'] = df['item_id'].astype(int)

    df_item = pd.DataFrame({'item_id': df['item_id'].unique()})
    for action_type in ['interaction', 'interaction item image', 'interaction item info', 
                        'interaction item rating', 'interaction item deals', 'search for item']:
        dff = df if action_type == 'interaction' else df[(df['action_type'] == action_type)]
        action_type_n = action_type.replace(' ', '_')

        df_item['item_%s_count' % action_type_n] = df_item['item_id'].map(
            dff.groupby('item_id')['session_id'].count()
        ).astype('float32')

        df_item['item_%s_session_count' % action_type_n] = df_item['item_id'].map(
            dff.groupby('item_id')['session_id'].nunique()
        ).astype('float32')

        df_item['item_%s_user_count' % action_type_n] = df_item['item_id'].map(
            dff.groupby('item_id')['user_id'].nunique()
        ).astype('float32')
            
    f_item_int_cnt.save(df_item)


@register(out=f_item_expo_cnt, inp=t_tr_te_flt_c_co)
def item_exposure_count():
    cols_keep = ['user_id', 'session_id', 'step', 'impressions']
    df_train_filter = t_tr_te_flt_c_co.train.load(columns=cols_keep)
    df_test_filter = t_tr_te_flt_c_co.test.load(columns=cols_keep)

    df_train_filter['session_id'] = df_train_filter['session_id'] + '_train'
    df_test_filter['session_id'] = df_test_filter['session_id'] + '_test'

    df = pd.concat([df_train_filter[cols_keep], df_test_filter[cols_keep]])
    df = df[~df['impressions'].isnull()]
    
    dff = explode(df[['user_id', 'session_id', 'step', 'impressions']], 'impressions')
    dff.rename(columns={'impressions': 'item_id'}, inplace=True)
    dff['item_id'] = dff['item_id'].astype(int)
        
    dff['index'] = np.arange(dff.shape[0])
    dff['rank'] = dff.groupby(['user_id', 'session_id', 'step'])['index'].rank()
    
    df_item = pd.DataFrame({'item_id': dff['item_id'].unique()})

    for num_keep in [None, 3, 5, 10]:
        dd = dff if num_keep is None else dff[dff['rank'] <= num_keep]
        postfix = '' if num_keep is None else '_top_%s' % num_keep

        df_item['item_exposure_count%s' % postfix] = df_item['item_id'].map(
            dd.groupby('item_id')['session_id'].count()
        ).astype('float32')

        df_item['item_exposure_session_count%s' % postfix] = df_item['item_id'].map(
            dd.groupby('item_id')['session_id'].nunique()
        ).astype('float32')

        df_item['item_exposure_user_count%s' % postfix] = df_item['item_id'].map(
            dd.groupby('item_id')['user_id'].nunique()
        ).astype('float32')
           
    f_item_expo_cnt.save(df_item) 


@register(out=f_sess_basic, inp=t_tr_te_target)
def session_basic():
    for tt in ['train', 'test']:
        target = t_tr_te_target[tt].load()

        cf = pd.DataFrame(list(target['current_filters'].str.split('|')\
                       .apply(lambda v: {} if not v else dict.fromkeys(v, 1))))
        cf.columns = ['filter_%s' % c for c in cf.columns]

        cols_keep_cf = ['filter_Sort By Distance', 'filter_Sort By Popularity', 'filter_Sort By Rating', 
                    'filter_Sort by Price', 'filter_Satisfactory Rating', 'filter_Very Good Rating', 
                    'filter_Focus on Distance', 'filter_Good Rating']
        for col in cols_keep_cf:
            target[col] = cf[col]

        target['ts'] = pd.to_datetime(target['timestamp'], unit='s')
        target['dt'] = target['ts'].dt.strftime('%Y%m%d')
        target['hour'] = target['ts'].dt.hour
        target['dayofweek'] = target['ts'].dt.dayofweek
        target['country'] = target['city'].str.split(',').str[-1]

        cols_keep = ['session_id', 'dt', 'hour', 'dayofweek', 'step', 
                     'platform', 'city', 'country', 'device',] + cols_keep_cf
        
        f_sess_basic[tt].save(target[cols_keep])


@register(out=f_sess_int, inp=t_tr_te_flt)
def session_interaction():
    for tt in ['train', 'test']:
        df = t_tr_te_flt[tt].load( 
            columns=['session_id', 'action_type', 'reference'])

        df_sess = pd.DataFrame({'session_id': df.session_id.unique()})
    
        df_sess['session_action_type_first'] = df_sess['session_id'].map(
            df.groupby('session_id')['action_type'].first()
        )
        df_sess['session_action_type_last'] = df_sess['session_id'].map(
            df.groupby('session_id')['action_type'].last()
        )

        dd = df[(~df['reference'].isnull()) & (df['reference'].str.isdigit())]
        action_types = ['interaction', 'interaction item image', 'interaction item info', 
                    'interaction item rating', 'interaction item deals', 
                    'search for item', 'clickout item']
    
        for at in action_types:
            dff = dd if at == 'interaction' else dd[dd.action_type == at]

            df_sess['session_%s_item_count' % at.replace(' ', '_')] = df_sess['session_id'].map(
                dff.groupby('session_id')['reference'].nunique()
            )

        df_sess = df_sess.reset_index(drop=True)
        f_sess_int[tt].save(df_sess)


@register(out=f_sess_imp_eq, inp=t_tr_te_flt_c_co)
def session_impressions_eq():
    for tt in ['train', 'test']:
        df = t_tr_te_flt_c_co[tt].load(
            columns=['session_id', 'action_type', 'step', 'step_target', 'impressions'])

        df_sess = pd.DataFrame({'session_id': df.session_id.unique()})

        df_sess['impressions_target'] = df_sess.session_id.map(
            df.groupby('session_id')['impressions'].last()
        )

        df = df[df['step'] < df['step_target']]
        df_sess['impressions_last'] = df_sess.session_id.map(
            df[['session_id', 'impressions']].groupby('session_id')
                .tail(2).groupby('session_id')['impressions'].first()
        )

        df_sess = df_sess[~df_sess['impressions_target'].isnull()]
        
        df_sess['session_impressions_eq_last'] = (df_sess['impressions_target'] == df_sess['impressions_last']) * 1
        df_sess.loc[df_sess['impressions_last'].isnull(), 'session_impressions_eq_last'] = np.nan

        df = df[~df['impressions'].isnull()]
        df_sess['impressions_last_imp'] = df_sess.session_id.map(
            df[['session_id', 'impressions']].groupby('session_id')
                .tail(2).groupby('session_id')['impressions'].first()
        )
  
        df_sess['session_impressions_eq_last_imp'] = (df_sess['impressions_target'] != df_sess['impressions_last_imp']) * 1
        df_sess.loc[df_sess['impressions_last_imp'].isnull(), 'session_impressions_eq_last_imp'] = np.nan
        
        df_sess = df_sess[['session_id', 'session_impressions_eq_last', 
                           'session_impressions_eq_last_imp']].reset_index(drop=True)
        f_sess_imp_eq[tt].save(df_sess)


@register(out=f_sess_imp, inp=t_tr_te_flt_c_co_imp)
def session_imp():
    for tt in ['train', 'test']:
        df = t_tr_te_flt_c_co_imp[tt].load(
            columns=['session_id', 'step', 'timestamp'])

        df_sess = pd.DataFrame({'session_id': df.session_id.unique()})

        df_sess['ts_max'] = df_sess.session_id.map(
            df.groupby('session_id')['timestamp'].max()
        )
        df_sess['ts_min'] = df_sess.session_id.map(
            df.groupby('session_id')['timestamp'].min()
        )
        df_sess['session_ts_all_imp'] = df_sess['ts_max'] - df_sess['ts_min']

        df_sess['step_max'] = df_sess.session_id.map(
            df.groupby('session_id')['step'].max()
        )
        df_sess['step_min'] = df_sess.session_id.map(
            df.groupby('session_id')['step'].min()
        )
        df_sess['session_step_all_imp'] = df_sess['step_max'] - df_sess['step_min']

        df_sess = df_sess[['session_id', 'session_ts_all_imp', 
                    'session_step_all_imp']].reset_index(drop=True)
        f_sess_imp[tt].save(df_sess)


@register(out=f_sess_int_price, inp=[t_tr_te_flt, t_tr_te_item_price])
def session_interaction_price():
    for tt in ['train', 'test']:
        df = t_tr_te_flt[tt].load( 
            columns=['session_id', 'reference'])
        df_price = t_tr_te_item_price[tt].load()

        df = df[(~df['reference'].isnull()) & (df['reference'].str.isdigit())]
        df.rename(columns={'reference': 'item_id'}, inplace=True)
        df['item_id'] = df['item_id'].astype(int)
        
        dff = df[['session_id', 'item_id']].drop_duplicates(['session_id', 'item_id'])
        dff = pd.merge(dff, df_price, on=['session_id', 'item_id'], how='inner')
        
        dd = dff.groupby('session_id')['session_item_price_mean'].agg([
            'first', 'last', 'min', 'max', 'mean', 'std']).reset_index()
        dd.rename(columns={'first': 'session_interaction_item_price_first', 
                           'last': 'session_interaction_item_price_last', 
                           'min': 'session_interaction_item_price_min', 
                           'max': 'session_interaction_item_price_max', 
                           'mean': 'session_interaction_item_price_mean', 
                           'std': 'session_interaction_item_price_std', }, inplace=True)
        
        dd['session_interaction_item_price_std'] = (dd['session_interaction_item_price_std'] / 
                                                    dd['session_interaction_item_price_mean'])
        f_sess_int_price[tt].save(dd)


@register(out=f_si_basic, inp=t_tr_te_target_expl)
def session_item_basic():
    for tt in ['train', 'test']:
        dd = t_tr_te_target_expl[tt].load(
                columns=['session_id', 'impression', 'price'])

        dd.rename(columns={'impression': 'item_id'}, inplace=True)
        dd['item_id'] = dd['item_id'].astype(int)

        dd['idx'] = np.arange(dd.shape[0])
        dd['item_rank'] = dd.groupby('session_id')['idx'].rank(method='first')
        dd['item_rank'] = dd['item_rank'] - 1
        
        dd['item_total'] = dd['session_id'].map(
            dd.groupby('session_id')['idx'].size()
        )
        dd['item_rank_ratio'] = 1.0 * dd['item_rank'] / dd['item_total']
        
        dd['price_rank'] = dd.groupby('session_id')['price'].rank(method='dense')
        dd['price_rank_max'] = dd['session_id'].map(
            dd.groupby('session_id')['price_rank'].max()
        )
        dd['price_rank_ratio'] = dd['price_rank'] / dd['price_rank_max']

        dd['price_rank_first'] = dd.groupby('session_id')['price'].rank(method='first')
        dd['price_rank_first_max'] = dd['session_id'].map(
            dd.groupby('session_id')['price_rank_first'].max()
        )
        dd['price_rank_first_ratio'] = dd['price_rank_first'] / dd['price_rank_first_max']

        cols_keep = ['session_id', 'item_id', 
                     'item_rank', 'item_total', 'item_rank_ratio', 
                     'price', 'price_rank', 'price_rank_first', 
                     'price_rank_ratio', 'price_rank_first_ratio']

        f_si_basic[tt].save(dd[cols_keep])


@register(out=f_si_first_last, inp=t_tr_te_flt_step)
def session_item_first_last():
    for tt in ['train', 'test']:
        df = t_tr_te_flt_step[tt].load(
            columns=['session_id', 'reference'])

        dff = df[(~df['reference'].isnull()) & (df['reference'].str.isdigit())]
        
        dff = dff.rename(columns={'reference': 'item_id'})
        dff['item_id'] = dff['item_id'].astype(int)
        
        dd_first = dff.groupby('session_id')['item_id'].first().reset_index()
        dd_first['session_item_first'] = 1

        dd_last = dff.groupby('session_id')['item_id'].last().reset_index()
        dd_last['session_item_last'] = 1

        dd = merge_all([dd_first, dd_last], 
                on=['session_id', 'item_id'], how='outer')
        f_si_first_last[tt].save(dd)


@register(out=f_si_int, inp=[t_tr_te_flt, t_tr_te_target])
def session_item_interaction():
    action_types = ['interaction', 'interaction item image', 'interaction item info', 
                    'interaction item rating', 'interaction item deals', 
                    'search for item', 'clickout item']

    for tt in ['train', 'test']:
        target = t_tr_te_target[tt].load(
            columns=['session_id', 'step', 'timestamp'])
#         target = target.drop_duplicates('session_id', keep='last')

        df = t_tr_te_flt[tt].load(
            columns=['session_id', 'reference', 'action_type', 'step', 'timestamp'])

        df = df[(~df['reference'].isnull()) & (df['reference'].str.isdigit())]
        df = df.rename(columns={'reference': 'item_id'})
        df['item_id'] = df['item_id'].astype(int)

        lst_dd = []
        for at in action_types:
            at_n = at.replace(' ', '_')

            dff = df if at == 'interaction' else df[df['action_type'] == at]

            series = dff.groupby(['session_id', 'item_id'])['action_type'].count()
            series.name = 'session_item_%s_count' % at_n
            lst_dd.append(series.reset_index())

            dd_first = dff[['session_id', 'item_id', 'step', 'timestamp']]\
                        .groupby(['session_id', 'item_id']).first()
            dd_first = dd_first.reset_index().rename(columns={
                'step': 'session_item_first_%s_step' % at_n, 
                'timestamp': 'session_item_first_%s_timestamp' % at_n 
            })
            lst_dd.append(dd_first)
            
            dd_last = dff[['session_id', 'item_id', 'step', 'timestamp']]\
                            .groupby(['session_id', 'item_id']).last()
            dd_last = dd_last.reset_index().rename(columns={
                'step': 'session_item_last_%s_step' % at_n, 
                'timestamp': 'session_item_last_%s_timestamp' % at_n 
            })

            lst_dd.append(dd_last)

        df_feat = merge_all(lst_dd, on=['session_id', 'item_id'], how='outer')
        df_feat = pd.merge(df_feat, target, on='session_id', how='left')

        for at in action_types:
            for fl in ['first', 'last']:
                for cc in ['step', 'timestamp']:
                    at_n = at.replace(' ', '_')
                    col = 'session_item_%s_%s_%s' % (fl, at_n, cc)
                    col_n = '%s_diff_target' % col
                    df_feat[col_n] = (df_feat[cc] - df_feat[col]).astype('float32')
        
        cols = ['session_item_interaction_item_image_count', 'session_item_interaction_item_info_count', 
                'session_item_search_for_item_count', 'session_item_clickout_item_count']

        df_feat['session_item_interaction_count'] = df_feat[cols].sum(axis=1)

        df_feat.drop(['step', 'timestamp'], axis=1, inplace=True)
        f_si_int[tt].save(df_feat)


@register(out=f_si_diff_last_int, inp=[t_tr_te_classify, t_tr_te_sess_last_int])
def session_item_diff_last_interaction_index():
    cols = ['session_last_interaction_item', 'session_last_interaction_item_image_item', 
            'session_last_interaction_item_info_item', 'session_last_interaction_item_rating_item', 
            'session_last_interaction_item_deals_item', 'session_last_search_for_item_item', 
            'session_last_clickout_item_item']
    for tt in ['train', 'test']:
        target = t_tr_te_classify[tt].load()
        target['index'] = np.arange(target.shape[0])

        df = t_tr_te_sess_last_int[tt].load()

        lst_dd = []
        for col in cols:
            col_t = '%s_index' % col
            col_n = '%s_index_diff' % col

            dd = df[['session_id', col]].rename(columns={col: 'item_id'})
            dd = dd[~dd['item_id'].isnull()]
            dd['item_id'] = dd['item_id'].astype(int)

            dd = pd.merge(target[['session_id', 'item_id', 'index']], dd, 
                          on=['session_id', 'item_id'], how='inner')
            dd = dd.rename(columns={'index': col_t})
            assert dd.shape[0] == dd.session_id.unique().shape[0]

            dd = pd.merge(target[['session_id', 'item_id', 'index']], 
                          dd[['session_id', col_t]], on='session_id', how='inner')
            dd[col_n] = dd['index'] - dd[col_t]

            dd['%s_gt_0' % col_n] = (dd[col_n] > 0) * 1
            dd['%s_gte_0' % col_n] = (dd[col_n] >= 0) * 1
            dd = dd[['session_id', 'item_id', col_n, 
                     '%s_gt_0' % col_n, '%s_gte_0' % col_n]]
            
            lst_dd.append(dd)

        df_feat = merge_all(lst_dd, on=['session_id', 'item_id'], how='outer')
        f_si_diff_last_int[tt].save(df_feat)


@register(out=f_si_diff_imp_price, inp=f_si_basic)
def session_item_price_div_imp_price():
    for tt in ['train', 'test']:
        df = f_si_basic[tt].load(
                columns=['session_id', 'item_id', 'price'])

        df_feat = df[['session_id', 'item_id', 'price']]

        df_feat['imp_price_first'] = df_feat.session_id.map(
            df.groupby('session_id')['price'].first()
        )

        cols_n = ['imp_price_first']
        for top_n in [None, 3, 5, 10]:
            dff = df if top_n is None else df.groupby('session_id').head(top_n)
            
            col_n = 'imp_price%s_min' % ('' if top_n is None else '_top_%s' % top_n)
            cols_n.append(col_n)

            df_feat[col_n] = df_feat.session_id.map(
                dff.groupby('session_id')['price'].min()
            )

        cols_keep = ['session_id', 'item_id']
        for col in cols_n:
            col_n = 'session_item_price_%s_div' % col
            df_feat[col_n] = df_feat['price'] / df_feat[col]
            cols_keep.append(col)
            cols_keep.append(col_n)
        
        f_si_diff_imp_price[tt].save(df_feat[cols_keep])


def label_encode(train, test, cols_le, cols_keep):
    from sklearn.preprocessing import LabelEncoder
    
    dd_train = train[cols_keep]
    dd_test = test[cols_keep]
    
    for col in cols_le:
        series = pd.concat([train[col].astype(str), 
                            test[col].astype(str)])
        le = LabelEncoder().fit(series)
        
        dd_train['%s_le' % col] = le.transform(train[col].astype(str))
        dd_test['%s_le' % col] = le.transform(test[col].astype(str))
       
    return dd_train, dd_test
        

@register(out=f_sess_le, inp=[f_sess_basic, f_sess_int])
def session_label_encode():
    cols_keep = ['session_id']
    cols_le = ['platform', 'city', 'country', 'device']

    df_train_b, df_test_b = label_encode(
        f_sess_basic.train.load(columns=cols_keep + cols_le),
        f_sess_basic.test.load(columns=cols_keep + cols_le),
        cols_le, cols_keep
    )
    
    cols_le = ['session_action_type_first', 'session_action_type_last']
    df_train_a, df_test_a = label_encode(
        f_sess_int.test.load(columns=cols_keep + cols_le),
        f_sess_int.test.load(columns=cols_keep + cols_le),
        cols_le, cols_keep
    )
    
    df_feat_train = pd.merge(df_train_b, df_train_a, on='session_id', how='outer')
    f_sess_le.train.save(df_feat_train)
    
    df_feat_test = pd.merge(df_test_b, df_test_a, on='session_id', how='outer')
    f_sess_le.test.save(df_feat_test)


rs_lgb_20190522_3 = [
    t_tr_te_classify,
    f_sess_basic,
    f_sess_imp,
    f_sess_int,
    f_sess_imp_eq,
    f_sess_int_price,
    f_sess_le,
    f_item_expo_cnt,
    f_item_int_cnt,
    f_item_meta_gte_50000,
    f_si_basic,
    f_si_diff_last_int,
    f_si_first_last,
    f_si_int,
    f_si_diff_imp_price,
]


def concat_lgb_20190522_3(tt):
    from feat_names import names_lgb_20190522_3 as feats

    cols_keep = ['session_id', 'item_id'] + feats 
    df = t_tr_te_classify[tt].load()

    df_sess = merge_all([
            f_sess_basic[tt].load(columns=cols_keep),
            f_sess_imp[tt].load(columns=cols_keep),
            f_sess_int[tt].load(columns=cols_keep),
            f_sess_imp_eq[tt].load(columns=cols_keep),
            f_sess_int_price[tt].load(columns=cols_keep),
            f_sess_le[tt].load(columns=cols_keep),
    ], on='session_id', how='outer')

    df_item = merge_all([
        f_item_expo_cnt.load(columns=cols_keep),
        f_item_int_cnt.load(columns=cols_keep),
        f_item_meta_gte_50000.load(columns=cols_keep),
    ], on='item_id', how='outer')

    df_sess_item = merge_all([
        f_si_basic[tt].load(columns=cols_keep),
        f_si_diff_last_int[tt].load(columns=cols_keep),
        f_si_first_last[tt].load(columns=cols_keep),
        f_si_int[tt].load(columns=cols_keep),
        f_si_diff_imp_price[tt].load(columns=cols_keep),
    ], on=['session_id', 'item_id'], how='outer')

    df = df.merge(df_sess, on='session_id', how='left')\
            .merge(df_item, on='item_id', how='left')\
            .merge(df_sess_item, on=['session_id', 'item_id'], how='left')

    for col in ['session_interaction_item_price_min', 'session_interaction_item_price_max', 
        'session_interaction_item_price_mean']:
        df['%s_div' % col] = df[col] / df['price']

    return df


@register(out=f_top30, inp=rs_lgb_20190522_3)
def top_30_feats():
    from feat_names import names_lgb_20190522_3_top_30 as feats_top_30

    cols_keep = ['session_id', 'item_id'] + feats_top_30
    cols_rn = ['session_id', 'impressions'] + ['m1_%s' % f for f in feats_top_30]

    for tt in ['train', 'test']:
        df = concat_lgb_20190522_3(tt)
        df = df[cols_keep]
        df.columns = cols_rn
        f_top30[tt].save(df)


@register(out=f_top100, inp=rs_lgb_20190522_3)
def top_100_feats():
    from feat_names import names_lgb_20190522_3_top_100 as feats_top_100

    cols_keep = ['session_id', 'item_id'] + feats_top_100
    cols_rn = ['session_id', 'impressions'] + ['m1_%s' % f for f in feats_top_100]

    for tt in ['train', 'test']:
        df = concat_lgb_20190522_3(tt)
        df = df[cols_keep]
        df.columns = cols_rn
        f_top100[tt].save(df)


@register(out=f_si_sim, inp=[t_tr_te_pair, t_sim])
def similarity_pair():
    cols_pair = [
        'item_id_impression_prev', 
    #     'item_id_impression_next', 
        'item_id_impression_first', 
    #     'item_id_interaction_first', 
        'item_id_interaction_last', 
        'item_id_interaction_most', 
    ]

    cols_sim = [
        'item_meta_cos',
        'co_appearence_interaction_count', 'co_appearence_impression_count',
        'similarity_wv_impression', 'similarity_wv_interaction',
    ]

    df_sim = t_sim.load(columns=['item_id', 'item_id_anchor'] + cols_sim)
    for tt in ['train', 'test']:
        df = t_tr_te_pair[tt].load()

        dd_lst = []
        for col in cols_pair:
            dd = pd.merge(df[['session_id', 'item_id', col]], df_sim, 
                          left_on=['item_id', col], right_on=['item_id', 'item_id_anchor'], how='left')
            dd = dd.rename(columns=dict([(c, '%s_%s' % (col, c)) for c in cols_sim]))
            dd.drop(['item_id_anchor', col], axis=1, inplace=True)
            dd_lst.append(dd)
            
        dd = merge_all(dd_lst, on=['session_id', 'item_id'], how='left')
        f_si_sim[tt].save(dd)


@register(out=f_si_cmp, inp=[t_tr_te_pair, f_top100])
def compare_pair():
    cols_pair = [
        'item_id_impression_prev', 
    #     'item_id_impression_next', 
        'item_id_impression_first', 
    #     'item_id_interaction_first', 
        'item_id_interaction_last', 
        'item_id_interaction_most', 
    ]

    cols_compare = [
        'price', 'price_rank', 'item_rank', 
        'session_item_interaction_count'
    ]

    for tt in ['train', 'test']:
        cols_keep = ['session_id', 'impressions'] + ['m1_%s' % f for f in cols_compare]
        cols_rn = ['session_id', 'item_id'] + cols_compare
        
        df = f_top100[tt].load(columns=cols_keep)
        df.columns = cols_rn
        
        df_pair = t_tr_te_pair[tt].load()
        df_pair = pd.merge(df_pair, df, on=['session_id', 'item_id'])
        
        dd_lst = []
        for col in cols_pair:
            cols_rn = dict([(c, '%s_anchor' % c) for c in cols_compare] + [('item_id', col)])
            dd_c = pd.merge(df_pair, df.rename(columns=cols_rn), on=['session_id', col])

            cols_keep = ['session_id', 'item_id']
            for cc in cols_compare:
                if cc == 'item_id':
                    continue

                col_n = '%s_%s_div' % (col, cc)
                dd_c[col_n] = (dd_c[cc] / dd_c[cols_rn[cc]]).replace(np.inf, -1)
                cols_keep.append(col_n)

            dd_lst.append(dd_c[cols_keep])

        dd = merge_all(dd_lst, on=['session_id', 'item_id'], how='outer')
        f_si_cmp[tt].save(dd)  
        

def collect_pair_win(df):
    df = df[['session_id', 'action_type', 'reference', 'impressions']]
    df = df[((df.action_type == 'clickout item') & (~df.reference.isnull()) & 
            (df.reference.str.isdigit()) & (~df.impressions.isnull()) & (df.impressions != ''))]
    
    dd = explode(df, 'impressions').rename(columns={'impressions': 'item_id_anchor', 'reference': 'item_id'})
    dd['item_id_anchor'] = dd['item_id_anchor'].astype(int)
    dd['item_id'] = dd['item_id'].astype(int)
    dd = dd[dd['item_id'] != dd['item_id_anchor']]
    
    return dd[['session_id', 'item_id', 'item_id_anchor']]


def te_pair_win(dd_source):
    dd = dd_source.groupby(['item_id', 'item_id_anchor'])['session_id'].size()
    dd = dd.reset_index().rename(columns={'session_id': 'item_win_counts'})
    
    dd = pd.merge(dd, dd.rename(columns={'item_id': 'item_id_anchor', 'item_id_anchor': 'item_id', 
                                         'item_win_counts': 'item_win_counts_reverse'}), 
                  on=['item_id', 'item_id_anchor'], how='left')
    
    dd_co = dd[~dd.item_win_counts_reverse.isnull()]
    
    dd_u = dd[dd.item_win_counts_reverse.isnull()]
    dd_r = dd_u.rename(columns={'item_id': 'item_id_anchor', 'item_id_anchor': 'item_id', 
                                'item_win_counts': 'item_win_counts_reverse', 'item_win_counts_reverse': 'item_win_counts'})
    
    dd = pd.concat([dd_co, dd_u, dd_r]).reset_index(drop=True)
    
    dd['item_win_counts'] = dd['item_win_counts'].fillna(0)
    dd['item_win_counts_reverse'] = dd['item_win_counts_reverse'].fillna(0)
    
    assert dd[dd.item_id == dd.item_id_anchor].shape[0] == 0
    assert dd.shape[0] == dd[['item_id', 'item_id_anchor']].drop_duplicates().shape[0]
    
    dd['item_win_ratio'] = dd['item_win_counts'] / (dd['item_win_counts'] + dd['item_win_counts_reverse'])
    
    return dd[['item_id', 'item_id_anchor', 'item_win_ratio']]


def merge_te_target(dd_te, dd_target):
    cols_pair = [
        'item_id_impression_prev', 
        'item_id_impression_first', 
        'item_id_interaction_last', 
        'item_id_interaction_most', 
    ]
    
    dd_lst = []
    
    for col in cols_pair:
        dd = pd.merge(dd_target[['session_id', 'item_id', col]], dd_te[['item_id', 'item_id_anchor', 'item_win_ratio']], 
                      left_on=['item_id', col], right_on=['item_id', 'item_id_anchor'], how='left')
        dd = dd.rename(columns={'item_win_ratio': '%s_item_win_ratio' % col})
        dd.drop(['item_id_anchor', col], axis=1, inplace=True)
        dd_lst.append(dd)
        
    dd = merge_all(dd_lst, on=['session_id', 'item_id'], how='left')
    return dd


@register(out=f_si_win, inp=[t_tr_te_flt_c_co, t_tr_te_pair, t_tr_te_classify])
def pair_win():
    cols_keep_df = ['session_id', 'action_type', 'reference', 'impressions']
    cols_keep_target = ['session_id', 'item_id', 'fold']
    
    df_train = t_tr_te_flt_c_co.train.load(columns=cols_keep_df)
    target_train = t_tr_te_classify.train.load(columns=cols_keep_target)
    df_pair_train = t_tr_te_pair.train.load()
    
    df_test = t_tr_te_flt_c_co.test.load(columns=cols_keep_df)
    target_test = t_tr_te_classify.test.load(columns=cols_keep_target)
    df_pair_test = t_tr_te_pair.test.load()
    
    df_win = collect_pair_win(df_train)
    
    dd_lst = []
    for fold_id in range(5):
        mask_source = df_win.session_id.isin(target_train[target_train['fold'] != fold_id].session_id.unique())
        dd_source = df_win[mask_source]
        
        dd_te = te_pair_win(dd_source)
        
        mask_target = df_pair_train.session_id.isin(target_train[target_train['fold'] == fold_id].session_id.unique())
        dd_target = df_pair_train[mask_target]
        
        dd = merge_te_target(dd_te, dd_target)
        dd_lst.append(dd)
        
    rr_train = pd.concat(dd_lst).reset_index(drop=True)
    
    dd_te = te_pair_win(df_win)
    rr_test = merge_te_target(dd_te, df_pair_test)
    
    f_si_win.train.save(rr_train) 
    f_si_win.test.save(rr_test)

