import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

def data_analysis(path):

	df = pd.read_csv(path)
	print(list(df))

	print(df['SeriousDlqin2yrs'].describe()) # target, skewed towards 0
	sns.countplot(x="SeriousDlqin2yrs", data=df)
	plt.show()

	print(df['RevolvingUtilizationOfUnsecuredLines'].describe()) # instances with ratios above 10 don't have normal defaults
	sns.displot(df["RevolvingUtilizationOfUnsecuredLines"])
	plt.show()
	print(df[df['RevolvingUtilizationOfUnsecuredLines'] > 10].SeriousDlqin2yrs.describe())

	print(df['age'].describe()) 
	sns.displot(df["age"])
	plt.show()

	print(df['NumberOfTime30-59DaysPastDueNotWorse'].describe())
	print(df['NumberOfTime60-89DaysPastDueNotWorse'].describe())
	print(df['NumberOfTimes90DaysLate'].describe()) # for values above 17, instances don't make sense, three columns share same value (96/98)
	print(df.groupby('NumberOfTimes90DaysLate').NumberOfTimes90DaysLate.count())

	print(df['DebtRatio'].describe()) # instances with values above the 97.5% percentile either don't have an income or (their income is 0/1 & their default is not normal)
	print(df.DebtRatio.quantile([0.975]))
	print(df[df['DebtRatio'] > 3489.025][['SeriousDlqin2yrs','MonthlyIncome']].describe())

	print(df['MonthlyIncome'].describe()) # missing values, replace with median, extreme outliers
	# num_null_vals = df.isnull().sum()

	print(df['NumberOfOpenCreditLinesAndLoans'].describe())
	sns.displot(df["NumberOfOpenCreditLinesAndLoans"])
	plt.show()

	print(df['NumberRealEstateLoansOrLines'].describe())
	sns.displot(df["NumberOfOpenCreditLinesAndLoans"])
	plt.show()

	print(df['NumberOfDependents'].describe()) # missing values, replace with mode (which is 0)
	print(df["NumberOfDependents"].value_counts())

	return

def main():
	training_path = 'data/cs-training.csv'
	data_analysis(path=training_path)

if __name__ == '__main__':
	main()