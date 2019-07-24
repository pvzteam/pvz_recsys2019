#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 基础模块
import os
import sys


if __name__ == '__main__':
    module = sys.argv[1]

    assert module in ('106', '107')

    if module == '106':
        print ('preprocess_flow')
        os.system('python -u preprocess_flow.py')

        lis = [0,1,3,6,8,9,14,15,20,22,24,26,27,35,36,37,38,41,64,67,71,82,87,105,106]
        for num in lis:
            print ('gen feat {}'.format(num))
            os.system('python -u feat{}.py'.format(num))

        print ('extract topk feat')
        os.system('python -u extract_topk_features.py')
    else:
        print ('gen feat {}'.format(107))
        os.system('python -u feat{}.py'.format(107))
