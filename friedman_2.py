import numpy as np
import math
import rvm_regression as rvm_r
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm

num_iterations = 100
training_samples = 240
test_samples = 1000
dimensions = 4

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
    training_targets[i] = pow(pow(X_training[i,0], 2) + pow(X_training[i,1] * 
    X_training[i,2] - 1 / (X_training[i,1] * X_training[i,3]), 2), 1/2)
    # Adding noise
    training_targets[i] += np.random.normal(0, training_targets[i]/3)

# alpha, variance_mp, mu_mp, sigma_mp = rvm_r.fit(X_training, variance, training_targets, kernel, training_samples)

# Generation of testing
# testing_targets = np.zeros(test_samples)
# for i in tqdm(range(num_iterations)):
#   for 
