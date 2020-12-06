import numpy as np 
import pandas as pd 
import joblib
import argparse

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier


def prepare_data(path, mode):
	df = pd.read_csv(path)
	if mode == 'train':
		df = df.drop(df[df['RevolvingUtilizationOfUnsecuredLines'] > 10].index) # remove RUUL outliers
		df = df.drop(df[df['DebtRatio'] > 3489.025].index) # remove DR outliers
		df = df.drop(df[df['NumberOfTimes90DaysLate'] > 17].index) # remove NT90DL outliers
	df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median()) # fill MI missing values with median
	df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0) # fill ND missing values with mode
	return df

def hyperparametr_search(dataset, model, target):
	features, label = dataset.drop([target], axis=1), dataset[target]
	params = {
		'learning_rate': [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.05, 0.1],
		'n_estimators': range(50, 750, 50),
		'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
		'max_depth': range(5, 10, 1),
		'alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1],
        'lambda': [0, 0.01, 0.05, 0.1, 0.5, 1],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'min_child_weight': [0, 1, 4, 7, 10]
		}
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
	return RandomizedSearchCV(model, param_distributions=params, n_iter=100, scoring='roc_auc', n_jobs=-1, cv=cv, verbose=3, random_state=0)

def train(dataset, model, target):
	features, label = dataset.drop([target], axis=1), dataset[target]
	model.fit(features, label, eval_metric='auc')
	return model

def test(dataset, model, target):
	features, label = dataset.drop([target], axis=1), dataset[target]
	return model.predict_proba(features)

def save_csv(dataset, result, fname):
	df = pd.DataFrame({"Id": dataset["Unnamed: 0"], "Probability": result})
	df["Id"] = df["Id"].astype(int)
	df["Probability"] = df["Probability"].astype(float)
	df.to_csv(fname, index=False) #save results

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--hps', type=bool, default=False, help='whether to do hyperparameter search or not')
	args = parser.parse_args()

	alg_name = 'xgboost'
	alg = XGBClassifier(random_state=0) # xgboost

	training_path = 'data/cs-training.csv'
	testing_path = 'data/cs-test.csv'
	target = 'SeriousDlqin2yrs' # target column

	if args.hps == True:
		print('Hyperparameter search... ')
		df_train = prepare_data(path=training_path, mode='train') #prepare training dataset
		model = hyperparametr_search(dataset=df_train, model=alg, target=target) #cross validation

		print('Training... ')
		trained_model = train(dataset=df_train, model=model, target=target) #fit model

		print('Saving trained model...')
		joblib.dump(trained_model, alg_name+'_model.p') #save model
	else:
		print('Using saved model... ')

	print('Testing... ')
	df_test = prepare_data(path=testing_path, mode='test') #prepare test dataset
	trained_model = joblib.load(alg_name+'_model.p')
	res = test(dataset=df_test, model=trained_model, target=target) #test model

	print('Saving results... ')
	save_csv(dataset=df_test, result=res[:,-1], fname=alg_name+'_results.csv') #save results

	print('Done!')

if __name__ == '__main__':
	main()

