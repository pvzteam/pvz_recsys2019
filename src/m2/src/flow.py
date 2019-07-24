#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 基础模块
import os
import sys
import time

ts = time.time()

num = sys.argv[1]
train_mode = 'norm'

if len(sys.argv) > 2:
    train_mode = sys.argv[2]

sub_file_path = '../model/lgb_s0_m2_{}'.format(num)
sub_name = 'lgb_s0_m2_{}'.format(num)

# lgb train
print ('normal lgb_train')
os.system('python -u lgb_train.py %s' % (num))

# merge cv & sub
print('\nkfold_merge')
os.system('python -u kfold_merge.py %s %s' % (sub_file_path, sub_name))

# convert cv & sub to list format
print ('\nconvert')
os.system('python -u convert.py %s %s' % (sub_file_path, sub_name))

# calculate mrr & auc
print ('\neval')
os.system('python -u eval.py %s' % ('{}/r_{}_cv.csv'.format(sub_file_path, sub_name)))

print ('all completed, cost {}s'.format(time.time() - ts))
