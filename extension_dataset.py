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
import itertools

# Initialize variable
N = 10
dimensions = 2
N_test = 100

X_1 = np.linspace(-10,10,N) # Training
X_2 = np.linspace(-10,10,N) # Training
X = np.zeros(((N)**2, 2))
for i, x in enumerate(itertools.product(X_1, X_2)):
    X[i, :] = x

X_1_test = np.linspace(-10,10,N_test) # Testing
X_2_test = np.linspace(-10,10,N_test) # Testing
X_test = np.zeros(((N_test)**2, 2))
for i, x in enumerate(itertools.product(X_1_test, X_2_test)):
    X_test[i, :] = x

variance = 0.01
tests = 100
pred_array = list() # Average regression

# Choose kernel between linear_spline or exponential
kernel = "exponential"

targets = np.zeros(X.shape[0])

for i in tqdm(range(X.shape[0])):
    targets[i] = math.sin(X[i, 0]) / X[i, 0] + 0.1 * X[i, 1] + np.random.normal(0, 0.1)

alpha, variance_mp, mu_mp, sigma_mp = rvm_r.fit(X, variance, targets, kernel, X.shape[0])
relevant_vectors = alpha[1].astype(int)

targets_test = np.zeros(X_test.shape[0])
y = np.zeros(X_test.shape[0])

for it in tqdm(range(tests)):
    for i in range(X_test.shape[0]):
        targets_test[i] =  math.sin(X_test[i, 0]) / X_test[i, 0] + 0.1 * X_test[i, 1] + np.random.normal(0, 0.1)
        y[i] = math.sin(X_test[i, 0]) / X_test[i, 0] + 0.1 * X_test[i, 1]
    pred_array.append(rvm_r.predict(X, X_test, relevant_vectors, variance_mp, mu_mp, sigma_mp, kernel, dimensions))
pred_mean = np.array(pred_array).mean(axis=0)
print('RMSE:', sqrt(mean_squared_error(y, pred_mean)))
print('Maximum error between predicted samples and true: ', max(abs(y-pred_mean))**2)
print('Number of relevant vectors:', len(relevant_vectors)-1)

sns.set(style="darkgrid")
fig = plt.figure()
fig.set_size_inches(15, 11)
ax = fig.gca(projection='3d')
ax.invert_xaxis()
ax.plot_trisurf(X_test[:,0], X_test[:,1], y, cmap=plt.cm.viridis, linewidth=0.2, label='True points')
ax.view_init(30, -50)
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('y')
plt.legend()
plt.show()

sns.set(style="darkgrid")
fig = plt.figure()
fig.set_size_inches(15, 11)
ax = fig.gca(projection='3d')
ax.invert_xaxis()
ax.scatter(X_test[:,0], X_test[:,1], pred_mean, cmap=plt.cm.viridis, linewidth=0.2, label='Training points')
ax.view_init(30, -50)
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('y')
plt.legend()
plt.show()