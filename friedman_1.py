import numpy as np
import math
import rvm_regression as rvm_r
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm

# Initialize variable
N = 240
N_pred = 1000
N_dimensions = 10
variance = 0.01
tests = 100

# Choose kernel between linear_spline or exponential
kernel = "linear_spline"

X_train = np.zeros((N, N_dimensions)) # Training
for i in range(N):
    for j in range(N_dimensions):
        X_train[i,j] = np.random.rand()

target_train = (10*np.sin(np.pi*X_train[:,0]*X_train[:,1])
                + 20*pow((X_train[:,2]
                - 1/2), 2) + 10*X_train[:,3]
                + 5*X_train[:,4]
                + np.random.uniform(-0.1, 0.1)
                )

X_pred_mean = []
for i in range(tests):
    X_pred = np.zeros((N_pred, N_dimensions)) # Test Prediction
    for j in range(N_pred):
        for k in range(N_dimensions):
            X_pred[j,k] = np.random.rand()
    
    X_pred_mean.append(X_pred)

X_pred = np.zeros((N_pred, N_dimensions)) # Test Prediction
for i in range(N_pred):
    X_pred[i,:] = X_pred_mean[:][0].mean(axis=0)

true_target = (10*np.sin(np.pi*X_pred[:,0]*X_pred[:,1])
                + 20*pow((X_pred[:,2]
                - 1/2), 2) + 10*X_pred[:,3]
                + 5*X_pred[:,4]
                )

# Fit
alpha, variance_mp, mu_mp, sigma_mp = rvm_r.fit(X_train, variance, target_train, kernel, N)
relevant_vectors = alpha[1].astype(int)

# Predict
target_pred = rvm_r.predict(X_train, X_pred, relevant_vectors, variance_mp, mu_mp, sigma_mp, kernel)

# Check Performance
print('RMSE:', sqrt(mean_squared_error(true_target, target_pred)))





