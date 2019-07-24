import pandas as pd
import numpy as np
import config

'''
https://github.com/recsyschallenge/2019/blob/master/src/baseline_algorithm/functions.py
'''

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
    df = pd.read_csv(ori)
    print(df.shape)
    df = df[df.action_type=='clickout item']
    print(df.shape)
    df_out = explode(df, ['impressions','prices'])
    print(df_out.shape)
    df_out.to_csv(des,index=False)

if __name__ == '__main__':
    gen_sample(config.data+'train.csv',config.data+'sample_train.csv')
    gen_sample(config.data+'test.csv',config.data+'sample_test.csv')
