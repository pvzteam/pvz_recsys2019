import pandas as pd
import sys
import utils
import cate_encoding
import config

def extract_star(x):
    ss = x.split('|')
    lis = []
    for s in ss:
        if 'Star' in s and 'From' not in s:
            lis.append(int(s[0]))
    if len(lis) == 0:
        return 0
    else:
        lis.sort()
        return lis[-1]

def extract_from_star(x):
    ss = x.split('|')
    lis = []
    for s in ss:
        if 'From' in s and 'Stars' in s:
            lis.append(int(s.split(' ')[1]))
    if len(lis) == 0:
        return 0
    else:
        lis.sort()
        return lis[-1]

def extract_rating(x):
    d = {'Satisfactory':1,'Good':2,'Very Good':3,'Excellent':4}
    ss = x.split('|')
    lis = []
    for s in ss:
        if 'Rating' in s:
            lis.append(d[s[:-7]])
    if len(lis) == 0:
        return 0
    else:
        lis.sort()
        return lis[-1]
    return 0

def gen_item_metafeat():
    df = pd.read_csv(config.data+'item_metadata.csv')
    df['feat_cnt'] = df['properties'].map(lambda x: len(x.split('|')))
    df['star'] = df['properties'].map(extract_star)
    df['from_star'] = df['properties'].map(extract_from_star)
    df['rating'] = df['properties'].map(extract_rating)
    df.drop('properties',axis=1,inplace=True)
    return df.rename(columns={'item_id':'impressions'})

def gen_item_impr_all():
    tr = pd.read_csv(config.data+'sample_train.csv',usecols = ['session_id','step','reference','impressions','prices'])
    te = pd.read_csv(config.data+'sample_test.csv',usecols = ['session_id','step','reference','impressions','prices'])
    df = pd.concat([tr,te])
    print(df.shape)
    df['impr_rank'] = df.groupby(['session_id','step']).cumcount().values
    df_impr = df.groupby(['impressions'])['step'].agg('count').reset_index()
    df_impr.rename(columns = {'step':'item_all_impr'},inplace=True)
    
    df_out = df[['impressions']].drop_duplicates()
    df_out = cate_encoding.cate_num_stat(df,df_out,['impressions'],'impr_rank',['median'])
    df_out = cate_encoding.cate_num_stat(df,df_out,['impressions'],'prices',['median'])
    df_feat = df_impr.merge(df_out,on='impressions',how='left')
    print(df_feat.head())
    return df_feat

def extract(df_ori, des):
    print(df_ori.shape)
    df_ori = df_ori.merge(df_meta, on = ['impressions'], how = 'left')
    df_ori = df_ori.merge(df_feat, on = ['impressions'], how = 'left')
    df_ori['item_price_div_median'] = df_ori['prices'] / df_ori['impressions_by_prices_median']
    df_ori['item_rank_sub_median'] = df_ori['impr_rank'] - df_ori['impressions_by_impr_rank_median']
    df_ori.drop(['impr_rank','prices'],axis=1,inplace=True)
    print(df_ori.head())
    df_ori.columns = df_ori.columns.astype(str)
    utils.save_df(df_ori, des)

if __name__ == '__main__':
    df_feat = gen_item_impr_all()
    df_meta = gen_item_metafeat()
    print(df_meta.head())

    trs = utils.load_df(config.feat+'m3_tr_0.ftr')
    tes = utils.load_df(config.feat+'m3_te_0.ftr')
    tr_out = trs[['session_id','impressions','impr_rank','prices']]
    te_out = tes[['session_id','impressions','impr_rank','prices']]
    extract(tr_out, config.feat+'m3_tr_item_feat.ftr')
    extract(te_out, config.feat+'m3_te_item_feat.ftr')




