import pandas as pd
import feather

def load_df(filename, nrows = None):
    if filename.endswith('csv'):
        return pd.read_csv(filename,nrows = nrows)
    elif filename.endswith('ftr'):
        #return pd.read_feather(filename)
        return feather.read_dataframe(filename)

def save_df( df, filename, index = False):
    if filename.endswith('csv'):
        df.to_csv(filename, index=index)
    elif filename.endswith('ftr'):
        df.to_feather(filename)

def reduce_mem(df_ori):
    for c in df_ori.columns:
        print(c)
        if c == 'impressions':
            continue
        if df_ori[c].dtype == 'int64':
            df_ori[c] = df_ori[c].astype('int32')
        if df_ori[c].dtype == 'float64':
            df_ori[c] = df_ori[c].astype('float32')
    return df_ori
