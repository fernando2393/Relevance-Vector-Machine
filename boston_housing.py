import numpy as np
import pandas as pd
import math
import rvm_regression as rvm_r
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Importing Boston Housing data set
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
y = boston["MEDV"].values
X = boston.drop(["MEDV"], axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Choose kernel between linear_spline or exponential
kernel = "linear_spline"

# Initialize variance
variance = 0.01
N = X_train.shape[0] # 481
dimensions = X_train.shape[1] #13

# Fit
alpha, variance_mp, mu_mp, sigma_mp = rvm_r.fit(X_train, variance, y_train, kernel, N)
relevant_vectors = alpha[1].astype(int)

# Predict
target_pred = rvm_r.predict(X_train, X_test, relevant_vectors, variance_mp, mu_mp, sigma_mp, kernel, dimensions)

# Check Performance
print('RMSE:', sqrt(mean_squared_error(y_test, target_pred)))
print('Number of relevant vectors: ', len(relevant_vectors))