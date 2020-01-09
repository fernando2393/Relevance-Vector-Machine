import numpy as np
import math
import rvm_regression as rvm_r
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

# Initialize variable
N = 100
N_test = 1000 
X = np.linspace(-10,10,N) # Training
X_test = np.linspace(-10,10, N_test) # Test
variance = 0.01
seed = 22
np.random.seed(seed)

# Choose kernel between linear_spline or exponential
kernel = "linear_spline"

#----- Case 1 -----#
targets = np.zeros(N)
targets_test = np.zeros(N_test)
y = np.zeros(len(X))
for i in range(len(X)):
    y[i] =  math.sin(X[i]) / X[i]
    targets[i] = math.sin(X[i]) / X[i] + np.random.uniform(-0.2, 0.2)

for i in range(N_test):
    targets_test[i] = math.sin(X_test[i]) / X_test[i] + np.random.uniform(-0.2, 0.2)

alpha, variance_mp, mu_mp, sigma_mp = rvm_r.fit(X, variance, targets, kernel, N)

relevant_vectors = alpha[1].astype(int)
prediction = rvm_r.predict(X, X_test, relevant_vectors, variance_mp, mu_mp, sigma_mp, kernel)

print('RMSE:', sqrt(mean_squared_error(targets_test, prediction)))

plt.plot(X_test, prediction, c='r', label='Predicted values')
plt.scatter(X, targets, label='Training samples')
plt.plot(X, y, c='black', label='True function')
plt.scatter(X[relevant_vectors[:-1]], targets[relevant_vectors[:-1]], c='r', marker='*', s=100)
plt.xlabel('X')
plt.ylabel('Target')
plt.legend()
plt.title('sinc(x) dataset with noise')
plt.show()