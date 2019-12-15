import pandas as pd
import numpy as np


def load_dataset(dataset):
	file = dataset + '.csv'
	dataset = pd.read_csv(file)
	dataset.replace('?',0, inplace=True)
	X = dataset.iloc[:,:-1].values
	Y = dataset.iloc[:,-1].values	
	return X, Y