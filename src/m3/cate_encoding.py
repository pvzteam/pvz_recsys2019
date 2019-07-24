import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
from  sklearn.linear_model import LinearRegression
import numpy as np
import category_encoders as ce

def label_encode(df, col):
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].fillna(-1))
    return df

def cate_num_stat(df_ori, df_des, group_cols, numerical, stat_methods, prefix = ''):
    df_stat = df_ori.groupby(group_cols)[numerical].agg(stat_methods).reset_index()
    for method in stat_methods:
        if not isinstance(method, str):
            method = method.__name__
        stat_name = prefix + '_'.join(group_cols) + '_by_' + numerical + '_' + method
        df_stat.rename(columns = { method : stat_name}, inplace = True)
    df_des = df_des.merge(df_stat, on = group_cols, how = 'left')
    return df_des

def cate_num_rank(df_ori, group_cols, numerical, ascending=True, show_agg = True):
    agg_name = ''
    if ascending:
        agg_name = '%s_by_%s_rank' % ('_'.join(group_cols) , numerical)
    if not ascending:
        agg_name = '%s_by_%s_revrank' % ('_'.join(group_cols) , numerical)
    if show_agg:
        print (agg_name)
    df_ori.sort_values([numerical], ascending = ascending,inplace=True)
    gp = df_ori.groupby(group_cols).cumcount()
    df_ori[agg_name] = gp.values
    return df_ori.sort_index()

'''
def cate_count_fast(df, group_cols, key_col, dtype = 'int64'):
    col_name = '{}_count'.format('_'.join(group_cols))
    print(col_name)
    df[col_name] = df.groupby(group_cols)[key_col].transform('count').astype(dtype)
    return df

def cate_count( df_ori, df_des, group_cols, agg_type='uint32', show_max=False, show_agg=True ):
    agg_name='{}_count'.format('_'.join(group_cols))
    if show_agg:
        print ("\nAggregating by ", group_cols ,  '... and saved in', agg_name)
    gp = df_ori[group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    #print gp
    df_des = df_des.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    if show_max:
        print (agg_name + " max value = ", df[agg_name].max())
    #print df_des[group_cols + [agg_name]]
    #df_des[agg_name] = df_des[agg_name].astype(agg_type)
    return df_des

def do_countuniq( df_ori,df_des, group_cols, counted, agg_type='uint32', show_agg=True ):
    agg_name= '{}_by_{}_countuniq'.format(('_'.join(group_cols)),(counted))
    if show_agg:
        print( "\nCounting unqiue ", counted, " by ", group_cols ,  '... and saved in', agg_name )
    gp = df_ori[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df_des = df_des.merge(gp, on=group_cols, how='left')
    del gp
    gc.collect()
    df_des[agg_name] = df_des[agg_name].astype(agg_type)
    return df_des

def cate_num_rank(df_ori, group_cols, numerical, ascending=True, show_agg = True):
    agg_name = ''
    if ascending:
        agg_name = '%s_by_%s_rank' % ('_'.join(group_cols) , numerical)
    if not ascending:
        agg_name = '%s_by_%s_revrank' % ('_'.join(group_cols) , numerical)
    if show_agg:
        print (agg_name)
    df_ori.sort_values([numerical], ascending = ascending,inplace=True)
    gp = df_ori.groupby(group_cols).cumcount()
    df_ori[agg_name] = gp.values
    return df_ori.sort_index()

def compute_trend(arr_y):
    try:
        x = np.arange(0,len(arr_y)).reshape(-1,1)
        lr = LinearRegression()
        lr.fit(x,arr_y)
        trend = lr.coef_[0]
    except:
        trend=np.nan
    return trend

def cate_num_trend(df_ori, df_des, group_cols, numerical, prefix = ''):
    print ("\n Group by ", group_cols ,  '... stat ',  numerical, 'features')
    df_stat = df_ori.groupby(group_cols)[numerical].agg(compute_trend).reset_index()
    print (df_stat.head())
    stat_name = prefix + '_'.join(group_cols) + '_by_' + numerical + '_trend'
    df_stat.rename(columns = { numerical : stat_name}, inplace = True)
    df_des = df_des.merge(df_stat, on = group_cols, how = 'left')
    return df_des

def cate_target_encoding(df_tr, df_te, group_cols, key, target, df_cv):
    te = cate_num_stat(df_tr[group_cols+[target]],df_te[group_cols+[key]],group_cols,target,['mean'])
    te.drop(group_cols, axis=1, inplace = True)

    df_tr = df_tr.merge(df_cv, on=key, how='left')
 
    tr_lis = []
    for cv in range(df_cv['cv'].max()+1):
        need_key = [key] + group_cols + [target]
        tr_tmp=[]
        mask = (df_tr.cv == cv)
        val_tr = df_tr.loc[mask]
        tra_tr = df_tr.loc[~mask]
        tr_tmp = cate_num_stat(tra_tr[group_cols+[target]],val_tr[group_cols+[key]],group_cols,target,['mean'])
        tr_tmp.drop(group_cols, axis=1, inplace = True)
        tr_lis.append(tr_tmp)
    tr = pd.concat(tr_lis,axis=0).reset_index(drop=True)
    return tr, te

def target_encode_all(df_tr, df_te, cols, target, min_samples_leaf=5):
    """https://github.com/scikit-learn-contrib/categorical-encoding/blob/master/category_encoders/target_encoder.py"""
    te = ce.TargetEncoder(cols=cols, return_df=True, min_samples_leaf=min_samples_leaf).fit(df_tr[cols], df_tr[target])
    feat_tr = te.transform(df_tr[cols], df_tr[target])
    feat_te = te.transform(df_te[cols])
    return feat_tr, feat_te

def cate_num_diff():
    return 0
'''
if __name__ == '__main__':
    df_tr = pd.read_csv('../../input/tr.csv')
    df_te = pd.read_csv('../../input/te.csv')
    df_cv = pd.read_csv('../../input/cv.csv')
    tr, te = cate_target_encoding(df_tr, df_te, ['group'], 'id', 'val', df_cv)

    print(df_tr)
    print(df_te)
    print(tr)
    print(te)
