# DS Challenge

## Contents

* `code.py`: Main script implementing data preparation, hyperparameter search, training and testing.
* `eda.py`: Exploratory data analysis done on the dataset.
* `xgboost_model.p`: Best trained model.
* `xgboost_results.csv`: Results on the test set based on the best trained model.
* `dsc.yml`: Environment file.
* `data`: Dataset.

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

## Approach
The overall approach is discussed below:

### Data preprocessing
1. An exploratory data analysis was conducted (`eda.py`). As a result, the following modifications were applied to the training set:
  * The entries for which `RevolvingUtilizationOfUnsecuredLines` was larger than 10, were dropped.
  * The entries for which `DebtRatio` was larger than 3489.025, were dropped.
  * The entries for which `NumberOfTimes90DaysLate` was larger than 17, were dropped.
  * Missing values for `MonthlyIncome` were filled with the median.
  * Missing values for `NumberOfDependents` were filled with mode.

2. Missing values for the test set were filled with the same approach.

### Model selection 
After exploring various models, **XGBoost** (eXtreme Gradient Boosting) which is an ensemble model, was chosen.

### Hyperparameter optimization
Out of several parameters for the XGBoost model, the following parameters were chosen for a randomized search along with a 5-fold cross validation:
* `learning_rate`
* `n_estimators`
* `subsample`
* `max_depth`
* `alpha`
* `lambda`
* `gamma`
* `min_child_weight`

After hyperparameter search, the best model is saved and then used for predicting the probabilities of the test set entries.

## Results
* Best cross-validation AUC score on the training set: **0.871**
* Private score on Kaggle: **0.866**
* Public score on Kaggle: **0.861**
