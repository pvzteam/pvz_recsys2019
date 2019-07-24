#!/bin/bash
set -e

## current path
pathwd=$(cd "$(dirname "$0")"; pwd)
cd $pathwd


## prepare similarity
cd $pathwd/src/m1
python3 main.py sim


## extract features
cd $pathwd/src/m1
python3 main.py feat

cd $pathwd/src/m2/src
python3 feature_flow.py 106

cd $pathwd/src/m3
./run.sh feat


## train individual models
cd $pathwd/src/m1
python3 main.py model

cd $pathwd/src/m2/src
python3 feature_flow.py 107
python3 flow.py 38
python3 flow.py 87
python3 flow.py 107


## stacking
cd $pathwd/src/m3
./run.sh model


## extract pairwise features 
## based on stacking prediction
cd $pathwd/src/m2/src
python3 feat108-pairwise.py 


## final train
cd $pathwd/src/m3
./run.sh final
