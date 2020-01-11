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
X_train = np.linspace(-10,10,N_train) # Training
X_test = np.linspace(-10,10, N_test) # Test
variance = 0.01
pred_array = list() # Average regression


# Choose kernel between linear_spline or exponential
kernel = "linear_spline"

#----- Case 1 -----#
y_train = np.zeros(N_train)
y_test = np.zeros(N_test)
y = np.zeros(N_test)

for i in range(len(X_train)):
    y_train[i] = math.sin(X_train[i]) / X_train[i] + np.random.uniform(-0.2, 0.2)

alpha, variance_mp, mu_mp, sigma_mp = rvm_r.fit(np.reshape(X_train,(N_train,dimensions)), variance, y_train, kernel, N_train)
relevant_vectors = alpha[1].astype(int)

for it in tqdm(range(tests)):
    for i in range(N_test):
        y[i] =  math.sin(X_test[i]) / X_test[i]
    pred_array.append(rvm_r.predict(X_train, X_test, relevant_vectors, variance_mp, mu_mp, sigma_mp, kernel, dimensions))
y_pred = np.array(pred_array).mean(axis=0)

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
plt.title('sinc(x) dataset with noise')
plt.show()

############## Comparisson with SVM from Scikit-Learn ##############

# Performance with SVM from sklearn
clf = svm.SVR(kernel=svm_methods.linear_spline)
clf.fit(np.reshape(X_train, (len(X_train), 1)), np.reshape(y_train, (len(y_train), 1)))
svm_pred = clf.predict(np.reshape(X_test, (len(X_test), 1)))
print('Number of support vectors:', len(clf.support_))
# Check Performance SVM
print('RMSE for SVM:', sqrt(mean_squared_error(y, svm_pred)))
plt.scatter(range(len(y)), y, label='Real')
plt.scatter(range(len(svm_pred)), svm_pred, c='orange', label='Predicted SVM')
plt.legend()
plt.show()