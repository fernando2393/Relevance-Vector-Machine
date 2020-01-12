import numpy as np
import math
import rvm_regression as rvm_r
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm
from sklearn.svm import SVR
from sklearn import svm
import itertools
import svm_methods

# Initialize variable
N_train = 10
dimensions = 2
N_test = 100
variance = 0.01

# Choose kernel between linear_spline, gaussian or exponential
kernel = "linear_spline"

X_1 = np.linspace(-10,10,N_train) # Training
X_2 = np.linspace(-10,10,N_train) # Training
X_train = np.zeros(((N_train)**2, 2))
for i, x in enumerate(itertools.product(X_1, X_2)):
    X_train[i, :] = x

y_train = np.zeros(X_train.shape[0])

for i in range(X_train.shape[0]):
    y_train[i] = math.sin(X_train[i, 0]) / X_train[i, 0] + 0.1 * X_train[i, 1] + np.random.normal(0,0.1)

X_1_test = np.linspace(-10,10,N_test) # Testing
X_2_test = np.linspace(-10,10,N_test) # Testing
X_test = np.zeros(((N_test)**2, 2))
for i, x in enumerate(itertools.product(X_1_test, X_2_test)):
    X_test[i, :] = x

alpha, variance_mp, mu_mp, sigma_mp = rvm_r.fit(X_train, variance, y_train, kernel, X_train.shape[0], dimensions, N_test)
relevant_vectors = alpha[1].astype(int)

y = np.zeros(X_test.shape[0])
for i in range(X_test.shape[0]):
    y[i] = math.sin(X_test[i, 0]) / X_test[i, 0] + 0.1 * X_test[i, 1]

y_pred = rvm_r.predict(X_train, X_test, relevant_vectors, variance_mp, mu_mp, sigma_mp, kernel, dimensions, N_test)

print('RMSE for RVM:', sqrt(mean_squared_error(y, y_pred)))
print('Maximum error between predicted samples and true: ', max(abs(y-y_pred))**2)
print('Number of relevant vectors:', len(relevant_vectors)-1)

sns.set(style="darkgrid")
fig = plt.figure()
fig.set_size_inches(15, 11)
ax = fig.gca(projection='3d')
ax.plot_trisurf(X_test[:,0], X_test[:,1], y_pred, cmap=plt.cm.plasma, linewidth=0.2)
ax.view_init(30, -50)
ax.scatter(X_train[:,0], X_train[:,1], y_train, c = "black", label='Training points')
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('y')
plt.title("Extension data set prediction")
plt.legend()
plt.show()

# Performance with SVM from sklearn
clf = svm.SVR(kernel='rbf')
clf.fit(np.reshape(X_train, (len(X_train), dimensions)), np.reshape(y_train, (len(y_train), 1)))
svm_pred = clf.predict(np.reshape(X_test, (len(X_test), dimensions)))
print('Number of support vectors:', len(clf.support_))
# Check Performance SVM
print('RMSE for SVM:', sqrt(mean_squared_error(y, svm_pred)))
sns.set(style="darkgrid")
fig = plt.figure()
fig.set_size_inches(15, 11)
ax = fig.gca(projection='3d')
ax.plot_trisurf(X_test[:,0], X_test[:,1], svm_pred, cmap=plt.cm.plasma, linewidth=0.2)
ax.view_init(30, -50)
ax.scatter(X_train[:,0], X_train[:,1], y_train, c = "black", label='Training points')
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('y')
plt.title("Extension data set prediction")
plt.legend()
plt.show()