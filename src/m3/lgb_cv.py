import sys
import utils
import pandas as pd
import os
import gc
import numpy as np
import lightgbm as lgb
import copy
import config

basic_param = {
    'boosting_type':'gbdt',
    'objective':'binary',     
    'metric':'auc',
    'learning_rate':0.1,
    'feature_fraction':0.7,
    'bagging_fraction':0.7,  
    'bagging_freq':2,
    'max_bin':255,
    'min_data_in_leaf':80,
    'min_child_weight':10,
    'lambda_l1':0,
    'lambda_l2':0,
    'max_depth':-1,
    'num_leaves':32,
    'verbose':0,
    'boost_from_average':False,
    'num_threads':36,
    'device':'cpu',
    'gpu_platform_id':0,
    'gpu_device_id' :0,
    'gpu_use_dp':False,
    'seed':2019
}

def train_cv(X_train,y_train,cv_id,feature_name,param, num_rounds, es_rounds, verbose_eval, metric, reverse, output, feval=None, weight = None, fobj=None):
    fold_num = cv_id.max() + 1
    d_scores = []
    d_preds = []
    for fold in range(fold_num):
        print('*' * 48)
        print('fold :' , fold)

        tr_idx, val_idx = (cv_id != fold).nonzero()[0], (cv_id == fold).nonzero()[0]
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        print(y_tr.mean())
        print(y_val.mean())
        if type(weight) == type(None):
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dvalid = lgb.Dataset(X_val, label=y_val)
        else:
            dtrain = lgb.Dataset(X_tr, label=y_tr, weight = weight[tr_idx])
            dvalid = lgb.Dataset(X_val, label=y_val, weight = np.array([1] * len(y_train)))

        valid_sets = [dvalid]
        valid_names = ['valid']
            
        evals_result = {}
        bst = lgb.train(param, dtrain, num_rounds, valid_sets, valid_names, early_stopping_rounds=es_rounds, verbose_eval=verbose_eval,evals_result = evals_result, feval=feval)
        bst.save_model(('%s/cv_%d.model') % (output,fold))
        d_preds.append(bst.predict(X_val,num_iteration=bst.best_iteration))
        d_scores.append(evals_result['valid'][metric][bst.best_iteration-1])


    # dump best score
    print(np.mean(d_scores))
    
    # cv_preds
    pred = np.zeros(X_train.shape[0])
    for fold in range(fold_num):
        val_idx = (cv_id == fold).nonzero()[0]
        pred[val_idx] = d_preds[fold]
    return pred

def predict_cv(test_X,output,fold_num):
    pred = np.zeros(test_X.shape[0])
    for i in range(fold_num):
        model = '%s/cv_%d.model' % (output,i)
        print (model)
        bst = lgb.Booster(model_file = model)
        pred += bst.predict(test_X)
    pred = pred / fold_num
    return pred

if __name__ == '__main__':
    idx = sys.argv[1]
    es = int(sys.argv[2])
    lr = float(sys.argv[3])
    nrows = None
    output = config.model + 'm3_%s' % (idx)
    if os.path.isdir(output) == False:
        os.mkdir(output)
    
    df_tr = utils.load_df(config.feat+'m3_tr_%s.ftr' % (idx))
    print(df_tr.head())
    print(df_tr.shape)
    df_cv = pd.read_csv(config.data+'cvid.csv')
    df_tr = df_tr.merge(df_cv, on = 'session_id', how='left')
    print(df_tr.shape)
    df_tr['cv'].fillna(-1,inplace=True)
    cvid = df_tr['cv'].astype(int)
    tr_Y = df_tr['target'].values
    df_key = df_tr[df_tr.cv >=0 ][['session_id','impressions','target']]
    tr_X = df_tr.drop(['target','timestamp','session_id','user_id','cv'],axis=1)
    feature_name = tr_X.columns
 
    tr_X = tr_X.values
    
    del df_tr
    gc.collect()
    
    basic_param['learning_rate']= lr
    print(basic_param)
     
    tr_pred = train_cv(tr_X,tr_Y,cvid,feature_name, basic_param, num_rounds = 10000, es_rounds = es, verbose_eval = 50, metric = 'auc', reverse = True,output = output)
    df_key['target'] = tr_pred[:df_key.shape[0]]
    df_key.to_csv('%s/tr_pred.csv' % output,index=False,float_format='%.4f')
    del tr_X
    gc.collect()
    
    df_te = utils.load_df(config.feat+'m3_te_%s.ftr' % (idx))
    print(df_te.shape)
    df_sub = pd.DataFrame()
    te_Y = df_te[feature_name].values

    te_pred = predict_cv(te_Y,output,5) 
    df_sub[['user_id','session_id','timestamp','step','impressions']] = df_te[['user_id','session_id','timestamp','step','impressions']]
    df_sub['target'] = te_pred
    df_sub[['session_id','impressions','target']].to_csv('%s/te_pred.csv' % output,index=False,float_format='%.4f')
    df_sub = df_sub.sort_values(by='target',ascending=False).reset_index(drop=True)
    df_sub['item_recommendations'] = df_sub['impressions'].astype(str)
    df_sub = df_sub.groupby(['user_id','session_id','timestamp','step'])['item_recommendations'].apply(lambda lst : ' '.join((lst))).reset_index()
    df_sub.to_csv('%s/sub.csv' % output,index=False)

    



    


