# The 3rd Place Solution for the 2019 ACM RecSys Challenge


## Introduction
This repository contains our solution for the 2019 ACM Recys Challenge.
Our solution consists of the following components. 
1. we cast this recommendation task as a binary classification problem.
2. we spend most of time on feature engineering and mining a series of useful features on various aspects. 
3. we train individual models with different set of features and blend them with some important features by stacking method. 
4. we create other new pairwise features based on existed model predictions and train a stacking model again which generate our final result.


## Environment
```
CentOS release 6.6 (Final)
python==3.6.6
numpy==1.15.4 
scikit-learn==0.20.2 
scipy==1.2.0
pandas==0.24.1
feather-format==0.4.0 
gensim==3.1.0
lightgbm==2.2.2
category-encoders==2.0.0
```


## Project Structure

```
├── input
├── src
├── output
├── tmp
├── feat
└── model
```

## Setup
Run the following command to install dependencies and create directories that conform to the structure of the project, then place the unzipped data into the ```input``` directory.:

```./setup.sh```


## Training & Submission
Simply run the following command to extract all the features generate the final results. The submission files are stored in the ```output``` directory. 
```
./run.sh
```


## Performance

| Model        | Local Validation AUC           | Local Validation MRR  |
| ------------- |-------------:| -----:|
| Best Single Model      | 0.9263 | 0.6822 |
| Stacking      | 0.9287      |   0.6851  |
| Stacking With Pairwise Features | 0.9310      |    0.6862  |

