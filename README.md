# DS Challenge

## Contents

* `code.py`: Main script implementing data preparation, hyperparameter search, training and testing.
* `eda.py`: Exploratory data analysis done on the dataset.
* `xgboost_model.p`: Best trained model.
* `xgboost_results.csv`: Results on the test set.

## How to run
1. Create an anaconda environment using the *yml* file included in the repository:
```Bash
conda create -f dsc.yml
```
2. Activate environment: 
```Bash
conda activate dsc
```
2. Run code (if hyperparameter search is desired, pass the `-hps` argument):
```Bash
python code.py 
```