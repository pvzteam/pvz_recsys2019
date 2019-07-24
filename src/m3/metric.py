import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import config
import utils

'''
https://github.com/recsyschallenge/2019/blob/master/src/baseline_algorithm/functions.py
'''

def get_reciprocal_ranks(ps):
    """Calculate reciprocal ranks for recommendations."""
    mask = ps.reference == np.array(ps.impressions)

    if mask.sum() == 1:
        rranks = generate_rranks_range(0, len(ps.impressions))
        return np.array(rranks)[mask].min()
    else:
        return 0.0
def generate_rranks_range(start, end):
    """Generate reciprocal ranks for a given list length."""

    return 1.0 / (np.arange(start, end) + 1)

tr = pd.read_csv(config.model + '%s/tr_pred.csv' % sys.argv[1])

tr = tr.sort_values(by='target',ascending=False).reset_index(drop=True)
tr = tr.groupby(['session_id'])['impressions'].apply(list).reset_index()
print(tr.head())
print(tr.shape)

tr_click = utils.load_df(config.data+'m3_tr_click.ftr')
tr = tr.merge(tr_click, on='session_id')
print(tr.shape)
print(tr.head())
tr['score'] = tr.apply(get_reciprocal_ranks, axis=1)
print(tr.score.mean())

