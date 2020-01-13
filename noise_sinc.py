import numpy as np
import math
import rvm_regression as rvm_r
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm
from sklearn import svm
import svm_methods

# Initialize variable
N_train = 100
N_test = 1000
tests = 100
dimensions = 1
variance = 0.01

# Choose kernel between linear_spline or exponential
kernel = "linear_spline"

X_train = np.linspace(-10,10,N_train) # Training
X_test = np.linspace(-10,10, N_test) # Test
y_train = np.zeros(N_train)
y = np.zeros(N_test)

for i in range(len(X_train)):
    y_train[i] = math.sin(X_train[i]) / X_train[i] + np.random.uniform(-0.2, 0.2)

alpha, variance_mp, mu_mp, sigma_mp = rvm_r.fit(np.reshape(X_train,(N_train,dimensions)), variance, y_train, kernel, N_train, dimensions, N_test)
relevant_vectors = alpha[1].astype(int)

print("Running RVM testing...")
for i in range(N_test):
    y[i] =  math.sin(X_test[i]) / X_test[i]
y_pred = rvm_r.predict(X_train, X_test, relevant_vectors, variance_mp, mu_mp, sigma_mp, kernel, dimensions, N_test)
print('RMSE:', sqrt(mean_squared_error(y, y_pred)))
print('Maximum error between predicted samples and true: ', max(abs(y - y_pred)) ** 2)
print('Number of relevant vectors:', len(relevant_vectors)-1)
plt.plot(X_test, y_pred, c='r', label='Predicted values')
plt.scatter(X_train, y_train, label='Training samples')
plt.plot(X_test, y, c='black', label='True function')
plt.scatter(X_train[relevant_vectors[:-1]], y_train[relevant_vectors[:-1]], c='r', marker='*', s=100, label='Relevant vectors')
plt.xlabel('X')
plt.ylabel('Target')
plt.legend()
plt.title('sinc(x) dataset with noise RVM')
plt.show()

############## Comparisson with SVM from Scikit-Learn ##############

# Performance with SVM from sklearn
if (kernel == "linear_spline"):
    clf = svm.SVR(kernel = svm_methods.linear_spline)
else:
    clf = svm.SVR(kernel="rbf", gamma = (1/dimensions))
clf.fit(np.reshape(X_train, (len(X_train), dimensions)), np.reshape(y_train, (len(y_train), dimensions)))
relevant_vectors = clf.support_
print("Running SVM testing...")
svm_pred = clf.predict(np.reshape(X_test, (len(X_test), dimensions)))
print('Number of support vectors:', len(clf.support_))
# Check Performance SVM
print('RMSE for SVM:', sqrt(mean_squared_error(y, svm_pred)))
plt.plot(X_test, svm_pred, c='r', label='Predicted values')
plt.scatter(X_train, y_train, label='Training samples')
plt.plot(X_test, y, c='black', label='True function')
plt.scatter(X_train[relevant_vectors[:-1]], y_train[relevant_vectors[:-1]], c='r', marker='*', s=100, label='Relevant vectors')
plt.legend()
plt.title('sinc(x) dataset with noise SVM')
plt.show()