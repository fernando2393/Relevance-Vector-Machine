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
test_array = list()

# Choose kernel between linear_spline or exponential
kernel = "gaussian"

# Generation of traning set
X_train = np.zeros((N_train, dimensions))
# Generation of samples
X_train[:,0] = np.random.uniform(0,100, N_train)
X_train[:,1] = np.random.uniform(40 * np.pi, 560 * np.pi, N_train)
X_train[:,2] = np.random.uniform(0,1, N_train)
X_train[:,3] = np.random.uniform(1,11, N_train)

# Generation of training targets
y_train = np.zeros(N_train)

for i in range(N_train):
    # Generating target without noise
    y_train[i] = pow(np.tan((X_train[i,1] * X_train[i,2] - 1 / (X_train[i,1] * X_train[i,3])
    ) / X_train[i,0]), -1)
    # Adding noise

y_train += np.random.normal(0, y_train.std()/3)

# Reshape to create scalar
y_train = np.reshape(y_train, (len(y_train),1))

# Scaling the dimensions to make proper comparisons
MinMaxScaler = preprocessing.MinMaxScaler()
X_train = MinMaxScaler.fit_transform(X_train)

alpha, variance_mp, mu_mp, sigma_mp = rvm_r.fit(X_train, variance, y_train, kernel, N_train, dimensions, N_test)
relevant_vectors = alpha[1].astype(int)
print("Number of relevant vectors:", len(relevant_vectors)-1)

# Generation of testing
X_test = np.zeros((N_test, dimensions))
y = np.zeros((N_test, tests))

print("Running RVM testing...")
for i in tqdm(range(tests)):
    X_test[:,0] = np.random.uniform(0,100, N_test)
    X_test[:,1] = np.random.uniform(40 * np.pi, 560 * np.pi, N_test)
    X_test[:,2] = np.random.uniform(0,1, N_test)
    X_test[:,3] = np.random.uniform(1,11, N_test)
    test_array.append(X_test)
X_test = np.array(test_array).mean(axis=0)
y = pow(np.tan((X_test[:,1] * X_test[:,2] - 1 / (X_test[:,1] * X_test[:,3])
    ) / X_test[:,0]), -1)
X_test = MinMaxScaler.fit_transform(X_test)
y_pred = rvm_r.predict(X_train, X_test, relevant_vectors, variance_mp, mu_mp, sigma_mp, kernel, dimensions, N_test)

print('RMSE for RVM:', sqrt(mean_squared_error(y, y_pred)))
plt.scatter(range(N_test), y, label='Real')
plt.scatter(range(N_test), y_pred, c='orange', label='Predicted RVM')
plt.legend()
plt.show()

# Performance with SVM from sklearn
clf = svm.SVR(kernel="rbf", gamma = (1/dimensions))
clf.fit(X_train, y_train)
print("Running SVM testing...")
svm_pred = clf.predict(np.reshape(X_test, (len(X_test), dimensions)))
print('Number of support vectors:', len(clf.support_))
# Check Performance SVM
print('RMSE for SVM:', sqrt(mean_squared_error(y, svm_pred)))
plt.scatter(range(len(y)), y, label='Real')
plt.scatter(range(len(svm_pred)), svm_pred, c='orange', label='Predicted')
plt.legend()
plt.show()