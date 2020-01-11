import numpy as np
import math
import rvm_regression as rvm_r
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm
from sklearn import preprocessing
from sklearn import svm
import svm_methods

tests = 100
N_train = 240
N_test = 1000
dimensions = 4
variance = 0.01
pred_array = list()

# Choose kernel between linear_spline or exponential
kernel = "linear_spline"

# Generation of traning set
X_train = np.zeros((N_train, dimensions))
# Generation of samples
X_train[:,0] = np.random.uniform(0,100, N_train)
X_train[:,1] = np.random.uniform(40 * np.pi, 560 * np.pi, N_train)
X_train[:,2] = np.random.uniform(0,1, N_train)
X_train[:,3] = np.random.uniform(1,11, N_train)

# Generation of training targets
y_train = np.zeros(N_train)

# Generating target without noise
y_train = pow(pow(X_train[:,0], 2) + pow(X_train[:,1] * 
X_train[:,2] - 1 / (X_train[:,1] * X_train[:,3]), 2), 1/2)
    
# Adding noise
for i in range(N_train):
    y_train[i] += np.random.normal(0, y_train.std()/3)

# Reshape to create scalar
y_train = np.reshape(y_train, (len(y_train),1))

# Scaling the dimensions to make proper comparisons
MinMaxScaler = preprocessing.MinMaxScaler()
X_train = MinMaxScaler.fit_transform(X_train)

alpha, variance_mp, mu_mp, sigma_mp = rvm_r.fit(X_train, variance, y_train, kernel, X_train.shape[0])
relevant_vectors = alpha[1].astype(int)
print("Number of relevant vectors:", len(relevant_vectors)-1)

# Generation of testing
X_test = np.zeros((N_test, dimensions))
y = np.zeros((N_test, tests))
for i in tqdm(range(tests)):
    X_test[:,0] = np.random.uniform(0,100, N_test)
    X_test[:,1] = np.random.uniform(40 * np.pi, 560 * np.pi, N_test)
    X_test[:,2] = np.random.uniform(0,1, N_test)
    X_test[:,3] = np.random.uniform(1,11, N_test)
    y[:,i] = pow(pow(X_test[:,0], 2) + pow(X_test[:,1] * 
    X_test[:,2] - 1 / (X_test[:,1] * X_test[:,3]), 2), 1/2)
    X_test = MinMaxScaler.fit_transform(X_test)
    pred_array.append(rvm_r.predict(X_train, X_test, relevant_vectors, variance_mp, mu_mp, sigma_mp, kernel, dimensions))

y = y.mean(axis=1)
pred_mean = np.array(pred_array).mean(axis=0)

print('RMSE for RVM:', sqrt(mean_squared_error(y, pred_mean)))
plt.scatter(range(N_test), y, label='Real')
plt.scatter(range(N_test), pred_mean, c='orange', label='Predicted RVM')
plt.legend()
plt.show()

# Performance with SVM from sklearn

clf = svm.SVR(kernel=svm_methods.linear_spline)
clf.fit(X_train, y_train)
svm_pred = clf.predict(X_test)
print('Number of support vectors:', len(clf.support_))
# Check Performance SVM
print('RMSE for SVM:', sqrt(mean_squared_error(y, svm_pred)))
plt.scatter(range(len(y)), y, label='Real')
plt.scatter(range(len(svm_pred)), svm_pred, c='orange', label='Predicted SVM')
plt.legend()
plt.show()