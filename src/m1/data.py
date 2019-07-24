# -*- coding: utf-8 -*-

import time
import numpy as np 
import pandas as pd 

from resource import *
from utils import (load_dataframe, explode, merge_all)
from utils import mock_name as _mm


i_tr_te = TrainTestResource(InputResource, '%s', 
                            fmt='csv', read_only=True)
i_item_metadata = InputResource('item_metadata', 
                                'csv', read_only=True)
i_cvid = InputResource('cvid', 'csv')

t_item_metadata_sp = TmpResource(_mm('item_metadata_splited'))

t_tr_te_target = TrainTestResource(TmpResource, _mm('target_%s'))
t_tr_te_target_expl = TrainTestResource(TmpResource, _mm('target_%s_expl'))
t_tr_te_classify = TrainTestResource(TmpResource, _mm('classify_%s'))

t_tr_te_flt = TrainTestResource(TmpResource, _mm('%s_filter'))
t_tr_te_flt_ts_c_co = TrainTestResource(TmpResource, _mm('%s_filter_contains_clickout_timestamp'))

t_tr_te_flt_step = TrainTestResource(TmpResource, _mm('%s_filter_step'))
t_tr_te_flt_c_co = TrainTestResource(TmpResource, _mm('%s_filter_contains_clickout'))

t_tr_te_flt_imp = TrainTestResource(TmpResource, _mm('%s_filter_imp'))
t_tr_te_flt_c_co_imp = TrainTestResource(TmpResource, _mm('%s_filter_contains_clickout_imp'))

t_tr_te_item_price = TrainTestResource(TmpResource, _mm('session_item_price_%s'))
t_tr_te_sess_last_int = TrainTestResource(TmpResource, _mm('session_last_interaction_item_%s'))

t_pair = TmpResource(_mm('pair_all'))
t_tr_te_pair = TrainTestResource(TmpResource, _mm('pair_%s'))

t_sim_item_meta_cos = TmpResource(_mm('similarity_item_meta_cos'))

t_sim_co_imp = TmpResource(_mm('similarity_co_appearence_impression'))
t_sim_co_int = TmpResource(_mm('similarity_co_appearence_interaction'))

t_sentence_imp = TmpResource(_mm('sentence_impression'), fmt='csv')
t_sentence_int = TmpResource(_mm('sentence_interaction'), fmt='csv')

t_sim_wv_imp = TmpResource(_mm('similarity_wv_impression'))
t_sim_wv_int = TmpResource(_mm('similarity_wv_interaction'))

t_sim = FeatResource(_mm('similarity_all'))


@register(out=t_item_metadata_sp, inp=i_item_metadata)
def split_item_metadata():
    df_item = i_item_metadata.load()

    dd = pd.DataFrame(list(df_item.properties.apply(lambda t: dict.fromkeys([v.lower() for v in t.split('|')], 1))), 
                      index=df_item.item_id).reset_index()

    t_item_metadata_sp.save(dd)


@register(out=t_tr_te_target, inp=i_tr_te)
def target_train_test():
    df = i_tr_te.train.load()
    mask = (~df["reference"].isnull()) & (df["action_type"] == "clickout item")
    target_val = df[mask].groupby(['user_id', 'session_id']).last().reset_index()

    t_tr_te_target.train.save(target_val)

    df = i_tr_te.test.load()
    mask = df["reference"].isnull() & (df["action_type"] == "clickout item")
    target_sub = df[mask].reset_index(drop=True)

    t_tr_te_target.test.save(target_sub)


@register(out=[t_tr_te_target_expl, t_tr_te_classify], 
          inp=[t_tr_te_target, i_cvid])
def explode_target_val_sub():
    def explode_impression_price(df):
        dd = explode(df, 'impressions').drop('prices', axis=1)
        dd_p = explode(df[['prices']], 'prices')
        return pd.concat([dd, dd_p], axis=1).rename(columns=
                {'impressions': 'impression', 'prices': 'price'})

    target_val = t_tr_te_target.train.load()
    target_val_expl = explode_impression_price(target_val)
    t_tr_te_target_expl.train.save(target_val_expl)

    classify_cv = target_val_expl[['user_id', 'session_id', 'reference', 'impression']]
    classify_cv = classify_cv.rename(columns={'impression': 'item_id'})
    classify_cv['item_id'] = classify_cv['item_id'].astype(int)
    classify_cv['reference'] = classify_cv['reference'].astype(int)
    classify_cv['target'] = (classify_cv['reference'] == classify_cv['item_id']) * 1
    
    cv_id = i_cvid.load()
    cv_id.rename(columns={'cv': 'fold'}, inplace=True)
    classify_cv = pd.merge(classify_cv, cv_id, on='session_id')
    assert classify_cv['fold'].isnull().sum() <= 0

    cols_keep = ['user_id', 'session_id', 'item_id', 'target', 'fold']
    t_tr_te_classify.train.save(classify_cv[cols_keep])


    target_sub = t_tr_te_target.test.load()
    target_sub_expl = explode_impression_price(target_sub)
    t_tr_te_target_expl.test.save(target_sub_expl)

    classify_sub = target_sub_expl[['user_id', 'session_id', 'reference', 'impression']]
    classify_sub = classify_sub.rename(columns={'impression': 'item_id'})
    classify_sub['item_id'] = classify_sub['item_id'].astype(int)
    classify_sub['target'] = 0.5

    cols_keep = ['user_id', 'session_id', 'item_id', 'target']
    t_tr_te_classify.test.save(classify_sub[cols_keep])

    
def filter_by_target(df, df_target, by, contains_clickout):
    col, col_n = by, '%s_target' % by
        
    df_target = df_target[['user_id', 'session_id', col]].rename(columns={col: col_n})
    df_filter = pd.merge(df, df_target, on=['user_id', 'session_id'], how='left')

    if contains_clickout is True:
        df_filter = df_filter[(~df_filter[col_n].isnull()) & 
                              (df_filter[col] <= df_filter[col_n])]
    else:
        df_filter = df_filter[(~df_filter[col_n].isnull()) & 
                              (df_filter[col] < df_filter[col_n])]

    df_filter = df_filter.reset_index(drop=True)
    return df_filter
   
    
@register(out=t_tr_te_flt_step, inp=[i_tr_te, t_tr_te_target])
def train_test_filter_step():
    for tt in ['train', 'test']:
        df = i_tr_te[tt].load()
        target = t_tr_te_target[tt].load()

        df_filter = filter_by_target(df, target, 'step', False)
        t_tr_te_flt_step[tt].save(df_filter)
        
        
@register(out=t_tr_te_flt, inp=[i_tr_te, t_tr_te_target])
def train_test_filter_timestamp():
    for tt in ['train', 'test']:
        df = i_tr_te[tt].load()
        target = t_tr_te_target[tt].load()

        df_filter = filter_by_target(df, target, 'timestamp', False)
        t_tr_te_flt[tt].save(df_filter)


@register(out=t_tr_te_flt_c_co, inp=[i_tr_te, t_tr_te_target])
def train_test_filter_contains_clickout_step():
    for tt in ['train', 'test']:
        df = i_tr_te[tt].load()
        target = t_tr_te_target[tt].load()

        df_filter = filter_by_target(df, target, 'step', True)
        t_tr_te_flt_c_co[tt].save(df_filter)

        
@register(out=t_tr_te_flt_ts_c_co, inp=[i_tr_te, t_tr_te_target])
def train_test_filter_contains_clickout_timestamp():
    for tt in ['train', 'test']:
        df = i_tr_te[tt].load()
        target = t_tr_te_target[tt].load()

        df_filter = filter_by_target(df, target, 'timestamp', True)
        t_tr_te_flt_ts_c_co[tt].save(df_filter)


def filter_by_impressions_begin(df):
    col, col_n = 'step', 'step_target_imp'
        
    action_types = ['filter selection', 'change of sort order', 'search for destination', 
                    'search for item', 'search for poi']
    df_target = df[df.action_type.isin(action_types)]

    df_target = df_target.groupby('session_id')[col].max().reset_index().rename(columns={col: col_n})

    df_filter = pd.merge(df, df_target, on=['session_id'], how='left')
    df_filter = df_filter[(df_filter[col_n].isnull()) |
                          (df_filter[col] >= df_filter[col_n])]
    df_filter = df_filter.reset_index(drop=True)
    
    return df_filter


@register(out=t_tr_te_flt_imp, inp=t_tr_te_flt)
def train_test_imp():
    for tt in ['train', 'test']:
        df = t_tr_te_flt[tt].load()
        df_imp = filter_by_impressions_begin(df)
        t_tr_te_flt_imp[tt].save(df_imp)


@register(out=t_tr_te_flt_c_co_imp, inp=t_tr_te_flt_ts_c_co)
def train_test_contains_clickout_imp():
    for tt in ['train', 'test']:
        df = t_tr_te_flt_ts_c_co[tt].load()
        df_imp = filter_by_impressions_begin(df)
        t_tr_te_flt_c_co_imp[tt].save(df_imp)


@register(out=t_tr_te_item_price, inp=t_tr_te_flt_c_co) 
def session_item_price():
    for tt in ['train', 'test']:
        df = t_tr_te_flt_c_co[tt].load()

        df = df[(~df['impressions'].isnull()) & (~df['prices'].isnull())]
        dd_i = explode(df[['session_id', 'impressions']], 'impressions')
        dd_p = explode(df[['prices']], 'prices')
        
        dd = pd.concat([dd_i, dd_p], axis=1).rename(columns={'impressions': 'item_id', 'prices': 'price'})
        dd['item_id'] = dd['item_id'].astype(int)
        dd['price'] = dd['price'].astype('float32')
        
        rr = dd.groupby(['session_id', 'item_id'])['price'].mean().reset_index()
        rr.rename(columns={'price': 'session_item_price_mean'}, inplace=True)
        
        t_tr_te_item_price[tt].save(rr)


@register(out=t_tr_te_sess_last_int, inp=t_tr_te_flt) 
def session_last_interaction_item():
    action_types = ['interaction', 'interaction item image', 'interaction item info', 
                    'interaction item rating', 'interaction item deals', 
                    'search for item', 'clickout item']

    for tt in ['train', 'test']:
        df = t_tr_te_flt[tt].load(
            columns=['session_id', 'reference', 'action_type'])
        dd = df[(~df['reference'].isnull()) & (df['reference'].str.isdigit())]
        
        dd = dd.rename(columns={'reference': 'item_id'})
        dd['item_id'] = dd['item_id'].astype(int)

        df_feat = pd.DataFrame({'session_id': dd['session_id'].unique()})

        for at in action_types:
            dff = dd if at == 'interaction' else dd[dd['action_type'] == at]

            df_feat['session_last_%s_item' % at.replace(' ', '_')] = df_feat['session_id'].map(
                dff.groupby('session_id')['item_id'].last()
            )
        
        t_tr_te_sess_last_int[tt].save(df_feat)


@register(out=t_tr_te_pair, inp=[t_tr_te_flt, t_tr_te_classify])
def pair_train_test():
    for tt in ['train', 'test']:
        df = t_tr_te_flt[tt].load(
            columns=['session_id', 'action_type', 'reference']
        )
        target = t_tr_te_classify[tt].load( 
            columns=['session_id', 'item_id']
        )

        dd_pair = target[['session_id', 'item_id']]

        dd_pair['item_id_impression_first'] = dd_pair.session_id.map(
            target.groupby('session_id')['item_id'].first()
        ).astype('float32')
    
        dd_pair['item_id_impression_prev'] = dd_pair.groupby('session_id')['item_id']\
                                                .shift(1).astype('float32')
        dd_pair['item_id_impression_next'] = dd_pair.groupby('session_id')['item_id']\
                                                .shift(-1).astype('float32')
        
        df_int = df[(~df['reference'].isnull()) & (df['reference'].str.isdigit())]
        df_int = df_int.rename(columns={'reference': 'item_id'})
        df_int['item_id'] = df_int['item_id'].astype(int)

        dd_pair['item_id_interaction_first'] = dd_pair.session_id.map(
            df_int.groupby('session_id')['item_id'].first()
        ).astype('float32')

        dd_pair['item_id_interaction_last'] = dd_pair.session_id.map(
            df_int.groupby('session_id')['item_id'].last()
        ).astype('float32')

        dd_int_most = df_int.groupby(['session_id', 'item_id'])['action_type'].size().reset_index()
        dd_int_most = dd_int_most.sort_values('action_type', ascending=False)

        dd_pair['item_id_interaction_most'] = dd_pair.session_id.map(
            dd_int_most.groupby('session_id')['item_id'].first()
        ).astype('float32')

        dd_pair = dd_pair.reset_index(drop=True)

        t_tr_te_pair[tt].save(dd_pair)
    

@register(out=t_pair, inp=[t_tr_te_flt, t_tr_te_classify, t_tr_te_pair])
def pair_all():
    cols_pair = [
        'item_id_impression_prev', 
        'item_id_impression_next', 
        'item_id_impression_first', 
        'item_id_interaction_first', 
        'item_id_interaction_last', 
        'item_id_interaction_most', 
    ]

    def transform(tt):
        df = t_tr_te_flt[tt].load(
            columns=['session_id', 'reference']
        )
        target = t_tr_te_classify[tt].load( 
            columns=['session_id', 'item_id']
        )

        df_pair = t_tr_te_pair[tt].load()

        df = df[(~df['reference'].isnull()) & (df['reference'].str.isdigit())]
        df = df.rename(columns={'reference': 'item_id_anchor'})
        df['item_id_anchor'] = df['item_id_anchor'].astype(int)
        
        rr = pd.merge(target, df, on='session_id')
        rr = rr[['item_id', 'item_id_anchor']].drop_duplicates()
        
        dd_lst = [rr]
        for col in cols_pair:
            dd = df_pair[['item_id', col]]
            dd = dd[~dd[col].isnull()]
            dd[col] = dd[col].astype(int)
            
            dd = dd.rename(columns={col: 'item_id_anchor'})
            dd_lst.append(dd)
        
        return pd.concat(dd_lst).drop_duplicates()

    df_pair_train = transform('train')
    df_pair_test = transform('test')

    df_pair = pd.concat([df_pair_train, df_pair_test])
    df_pair = df_pair.drop_duplicates().reset_index(drop=True)
    df_pair['item_id'] = df_pair['item_id'].astype(int)
    df_pair['item_id_anchor'] = df_pair['item_id_anchor'].astype(int)

    t_pair.save(df_pair)


@register(out=t_sim_item_meta_cos, inp=[t_item_metadata_sp, t_pair])
def similarity_item_meta_cos():
    df_meta = t_item_metadata_sp.load()
    df_meta = df_meta.fillna(0).astype(int)

    cols_meta = [c for c in df_meta.columns if c != 'item_id']

    df_pair = t_pair.load()
    df_pair = df_pair.reset_index()

    rr = np.zeros(df_pair.shape[0]).astype('float32') * np.nan

    batch_size = 1000000
    for i in range(int(np.ceil(df_pair.shape[0] / batch_size))):
        s, e = i * batch_size, (i + 1) * batch_size
        dd = df_pair.iloc[s:e]
        
        if dd.shape[0] <= 0:
            break
            
        arr_item = pd.merge(dd, df_meta, on='item_id', how='left').sort_values('index')
        arr_item_anchor = pd.merge(dd, df_meta, left_on='item_id_anchor', 
                        right_on='item_id', how='left').sort_values('index')[cols_meta]

        rr[s:e] = ((arr_item[cols_meta] * arr_item_anchor[cols_meta]).sum(axis=1) / 
                        (arr_item.sum(axis=1) ** 0.5) / (arr_item_anchor.sum(axis=1) ** 0.5))
        
    df_pair['item_meta_cos'] = rr
    t_sim_item_meta_cos.save(df_pair[['item_id', 'item_id_anchor', 'item_meta_cos']])


@register(out=t_sim_co_int, inp=[t_tr_te_flt, t_pair])
def similarity_co_appearence_interaction():
    def transform(tt):
        df = t_tr_te_flt[tt].load(columns=['session_id', 'reference'])
        df = df[(~df['reference'].isnull()) & (df['reference'].str.isdigit())]
        df = df.rename(columns={'reference': 'item_id'})
        df['item_id'] = df['item_id'].astype(int)

        df = df.drop_duplicates()
        dd = pd.merge(df, df.rename(columns={'item_id': 'item_id_anchor'}), on='session_id')
        dd = dd[dd.item_id != dd.item_id_anchor]

        dd = dd.groupby(['item_id', 'item_id_anchor'])['session_id'].size().reset_index()
        dd = dd.rename(columns={'session_id': 'co_appearence_interaction_count'})

        return dd

    dd_train = transform('train')
    dd_test = transform('test')

    dd = pd.concat([dd_train, dd_test])
    dd = dd.groupby(['item_id', 'item_id_anchor'])['co_appearence_interaction_count'].sum()

    df_pair = t_pair.load()

    df_pair = pd.merge(df_pair, dd.reset_index(), on=['item_id', 'item_id_anchor']).reset_index(drop=True)
    t_sim_co_int.save(df_pair)


@register(out=t_sim_co_imp, inp=[t_tr_te_flt_c_co, t_pair])
def similarity_co_appearence_impression():
    def transform(tt):
        df = t_tr_te_flt_c_co[tt].load( 
                columns=['session_id', 'step', 'impressions']
        )
        df = df[(~df['impressions'].isnull()) & (df['impressions'] != '')]
        df = df.drop_duplicates(['session_id', 'impressions'], keep='last')

        df = explode(df, 'impressions')
        df.rename(columns={'impressions': 'item_id'}, inplace=True)
        df['item_id'] = df['item_id'].astype(int)
        
        df = df.drop_duplicates()
        
        dd = pd.merge(df, df.rename(columns={'item_id': 'item_id_anchor'}), on=['session_id', 'step'])
        dd = dd[dd.item_id != dd.item_id_anchor]
        
        dd = dd.groupby(['item_id', 'item_id_anchor'])['session_id'].size()
        dd = dd.reset_index().rename(columns={'session_id': 'co_appearence_impression_count'})

        return dd

    dd_train = transform('train')
    dd_test = transform('test')

    dd = pd.concat([dd_train, dd_test])
    dd = dd.groupby(['item_id', 'item_id_anchor'])['co_appearence_impression_count'].sum()

    df_pair = t_pair.load()

    df_pair = pd.merge(df_pair, dd.reset_index(), on=['item_id', 'item_id_anchor']).reset_index(drop=True)
    t_sim_co_imp.save(df_pair)


@register(out=t_sentence_int, inp=t_tr_te_flt)
def sentence_interaction():
    def transform(tt):
        df = t_tr_te_flt[tt].load( 
                columns=['session_id', 'action_type', 'reference']
        )

        dd = df.drop_duplicates()

        dd.loc[:, 'word'] = ''

        action_types = ['clickout item', 'interaction item image', 
            'interaction item info', 'interaction item rating', 
            'interaction item deals']

        dd.loc[(dd.action_type.isin(set(action_types)) & 
            (~dd['reference'].isnull()) & 
            (dd['reference'].str.isdigit())), 'word'] = dd['reference'].astype(str)

        action_types = ['filter selection', 'search for destination', 
            'change of sort order', 'search for item', 
            'search for poi']
        dd.loc[(dd.action_type.isin(set(action_types)) & 
                (~dd['reference'].isnull()) & 
                (dd['reference'] != '')), 'word'] = (
            dd.action_type.str.replace('[ ,"\']', '_') + '_' + 
            dd['reference'].astype(str).str.replace('[ ,"\']', '_')
        )

        ss = dd.groupby('session_id')['word'].size()
        ss = ss[ss <= 1]
        ddf = dd[~dd.session_id.isin(ss.index)]

        rs = ddf[['session_id', 'word']].groupby('session_id')['word'].apply(
                lambda x: ' '.join(map(str, x))).reset_index()
        rs.rename(columns={'word': 'sentence'}, inplace=True)
        
        return rs

    df_sentence_train = transform('train')
    df_sentence_test = transform('test')

    df_sentence = pd.concat([df_sentence_train[['sentence']], 
                             df_sentence_test[['sentence']]])

    df_sentence.to_csv(t_sentence_int.path, '\t', index=False, header=False)


@register(out=t_sentence_imp, inp=t_tr_te_flt_c_co)
def sentence_impression():
    def transform(tt):
        df = t_tr_te_flt_c_co[tt].load( 
            columns=['session_id', 'impressions']
        )

        dd = df[~df['impressions'].isnull()].drop_duplicates()
        dd['sentence'] = dd['impressions'].str.replace('\|', ' ')
        
        return dd[['sentence']]

    df_sentence_train = transform('train')
    df_sentence_test = transform('test')

    df_sentence = pd.concat([df_sentence_train[['sentence']], 
                             df_sentence_test[['sentence']]])

    df_sentence.to_csv(t_sentence_imp.path, '\t', index=False, header=False)


@register(out=t_sim_wv_imp, inp=[t_sentence_imp, t_pair])
def similarity_wv_impression():
    import gensim

    sentences = gensim.models.word2vec.LineSentence(
        t_sentence_imp.path)

    model = gensim.models.Word2Vec(sentences, size=25, window=5, 
        min_count=2, iter=20)

    df_pair = t_pair.load()
    vv = df_pair.values

    rr = np.zeros(df_pair.shape[0]).astype('float32') * np.nan

    for i in range(df_pair.shape[0]):
        try:
            rr[i] = model.similarity(str(vv[i, 0]), str(vv[i, 1]))
        except Exception as e:
            pass

    df_pair['similarity_wv_impression'] = rr
    t_sim_wv_imp.save(df_pair)


@register(out=t_sim_wv_int, inp=[t_sentence_int, t_pair])
def similarity_wv_interaction():
    import gensim

    sentences = gensim.models.word2vec.LineSentence(
        t_sentence_int.path)

    model = gensim.models.Word2Vec(sentences, size=25, window=5, 
        min_count=2, iter=20)

    df_pair = t_pair.load()
    vv = df_pair.values

    rr = np.zeros(df_pair.shape[0]).astype('float32') * np.nan

    for i in range(df_pair.shape[0]):
        try:
            rr[i] = model.similarity(str(vv[i, 0]), str(vv[i, 1]))
        except Exception as e:
            pass

    df_pair['similarity_wv_interaction'] = rr
    t_sim_wv_int.save(df_pair)


@register(out=t_sim, inp=[
        t_pair, t_sim_item_meta_cos, 
        t_sim_co_imp, t_sim_co_int, 
        t_sim_wv_imp, t_sim_wv_int
    ])
def similarity_all():
    df_pair = merge_all([
        t_pair.load(),
        t_sim_item_meta_cos.load(),
        t_sim_co_imp.load(),
        t_sim_co_int.load(),
        t_sim_wv_imp.load(),
        t_sim_wv_int.load(),
    ], on=['item_id', 'item_id_anchor'], how='left')

    t_sim.save(df_pair)


