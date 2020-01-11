import numpy as np
import math
import rvm_regression as rvm_r
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm
from sklearn import preprocessing
from sklearn import svm

num_iterations = 100
training_samples = 240
test_samples = 1000
dimensions = 4
variance = 0.01
pred_array = list()

# Choose kernel between linear_spline or exponential
kernel = "linear_spline"

# Generation of traning set
X_training = np.zeros((training_samples, dimensions))
# Generation of samples
X_training[:,0] = np.random.uniform(0,100, training_samples)
X_training[:,1] = np.random.uniform(40 * np.pi, 560 * np.pi, training_samples)
X_training[:,2] = np.random.uniform(0,1, training_samples)
X_training[:,3] = np.random.uniform(1,11, training_samples)

# Generation of training targets
training_targets = np.zeros(training_samples)

for i in range(training_samples):
    # Generating target without noise
    training_targets[i] = pow(np.tan((X_training[i,1] * X_training[i,2] - 1 / (X_training[i,1] * X_training[i,3])
    ) / X_training[i,0]), -1)
    # Adding noise

training_targets += np.random.normal(0, training_targets.std()/3)

# Reshape to create scalar
training_targets = np.reshape(training_targets, (len(training_targets),1))

# Scaling the dimensions to make proper comparisons
MinMaxScaler = preprocessing.MinMaxScaler()
X_training = MinMaxScaler.fit_transform(X_training)

alpha, variance_mp, mu_mp, sigma_mp = rvm_r.fit(X_training, variance, training_targets, kernel, training_samples)
relevant_vectors = alpha[1].astype(int)
print("Number of relevant vectors:", len(relevant_vectors)-1)

# Generation of testing
X_test = np.zeros((test_samples, dimensions))
y = np.zeros((test_samples, num_iterations))
for i in tqdm(range(num_iterations)):
    X_test[:,0] = np.random.uniform(0,100, test_samples)
    X_test[:,1] = np.random.uniform(40 * np.pi, 560 * np.pi, test_samples)
    X_test[:,2] = np.random.uniform(0,1, test_samples)
    X_test[:,3] = np.random.uniform(1,11, test_samples)
    y[:,i] = pow(np.tan((X_test[:,1] * X_test[:,2] - 1 / (X_test[:,1] * X_test[:,3])
    ) / X_test[:,0]), -1)
    X_test = MinMaxScaler.fit_transform(X_test)
    pred_array.append(rvm_r.predict(X_training, X_test, relevant_vectors, variance_mp, mu_mp, sigma_mp, kernel, dimensions))

y = y.mean(axis=1)
pred_mean = np.array(pred_array).mean(axis=0)

print('RMSE:', sqrt(mean_squared_error(y, pred_mean)))
plt.plot(range(test_samples), y)
plt.scatter(range(test_samples), pred_mean, c='orange')
plt.show()

# Performance with SVM from sklearn
clf = svm.SVR()
clf.fit(X_training, training_targets)
svm_predict = clf.predict(X_test)
print('Number of support vectors:', len(clf.support_vectors_))
# Check Performance SVM
print('RMSE for SVM:', sqrt(mean_squared_error(y, svm_predict)))
plt.scatter(range(test_samples), y, label='Real')
plt.scatter(range(test_samples), svm_predict, c='orange', label='Predicted')
plt.legend()
plt.show()