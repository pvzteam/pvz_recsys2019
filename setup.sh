#!/bin/bash
set -e

## current path
pathwd=$(cd "$(dirname "$0")"; pwd)
cd $pathwd

pip install -U pip

pip install 'numpy>=1.15.4'
pip install 'scipy>=1.2.0'
pip install 'pandas>=0.24.1'
pip install 'scikit-learn>=0.20.2'
pip install 'feather-format>=0.4.0 '
pip install 'gensim>=3.1.0'
pip install 'lightgbm>=2.2.2'
pip install category_encoders


mkdir output tmp feat model

ln -s $pathwd/input $pathwd/src/m2/
ln -s $pathwd/feat $pathwd/src/m2/feature
ln -s $pathwd/model $pathwd/src/m2/
