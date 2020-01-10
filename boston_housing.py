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
from sklearn.preprocessing import StandardScaler


# Importing Boston Housing data set
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
y = boston["MEDV"].values
X = boston.drop(["MEDV"], axis = 1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Reshape to create scalar
y_train = np.reshape(y_train, (len(y_train),1))

# Scaling the dimensions to make proper comparisons 
scaleX = StandardScaler().fit(X_train)
scaley = StandardScaler().fit(y_train)
X_train = scaleX.transform(X_train)
y_train = scaley.transform(y_train)
X_test = scaleX.transform(X_test)



# Choose kernel between linear_spline or exponential
kernel = "linear_spline"

# Initialize variance
variance = 0.01
N = X_train.shape[0] # 480
dimensions = X_train.shape[1] #14
N_test_size = X_test.shape[0]

# Fit
alpha, variance_mp, mu_mp, sigma_mp = rvm_r.fit(X_train, variance, y_train, kernel, N)
relevant_vectors = alpha[1].astype(int)

# Predict
target_pred = rvm_r.predict(X_train, X_test, relevant_vectors, variance_mp, mu_mp, sigma_mp, kernel, dimensions)

# Inverse trasmform to real Scale
y_pred = scaley.inverse_transform(target_pred)

# Check Performance
print('RMSE:', sqrt(mean_squared_error(y_test, y_pred)))
print('Number of relevant vectors: ', len(relevant_vectors))