import numpy as np
import math
import rvm_regression as rvm_r
import matplotlib.pyplot as plt


X = np.linspace(-10,10,100) # Training
variance = 0.01
#----- Case 1 -----#

#aux = [np.random.normal(0, pow(variance,2)) for i in range(1000)]
targets = np.zeros(len(X))
for i in range(len(X)):
    targets[i] = math.sin(X[i])/X[i] + np.random.normal(0, pow(variance,2))

alpha, variance_mp, mu_mp, sigma_mp, Basis_mp = rvm_r.fit(X, variance, targets)
X_test = np.linspace(-10,10,100) # Test
prediction = rvm_r.predict(X_test, alpha, variance_mp, mu_mp, sigma_mp, Basis_mp)
#print("Predicted:", prediction)
plt.scatter(X, targets)
plt.scatter(X_test, prediction)
plt.show()

#----- Case 2 -----#
"""
aux += np.random.uniform(-0.1, 0.1, 100)
targets = np.zeros(len(X))
for i in range(len(X)):
    targets[i] = math.sin(X[i])/X[i] + (1/1000) * np.sum(aux)

X_test = np.linspace(-10,10,100) # Test
"""