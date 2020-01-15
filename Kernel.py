import math

import numpy as np


# All defined kernels are defined in this class
from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances

def generalized_t_student_kernel(X, Y):
    d = X.shape[0]
    X_aux = abs(X)
    Y_aux = abs(Y)
    compute_kernel = 1 / (1 - pow(euclidean_distances(X_aux, Y_aux, squared=True), d))
    return compute_kernel

def combination_spherical_t_student_kernel(X, Y, r=None):
    if r is None:
        r = X.shape[1]
    compute_kernel = 1 - (3/2) * euclidean_distances(X, Y) / r + (1/2) * pow(euclidean_distances(X, Y) / r, 3) # Spherical
    d = X.shape[0]
    X_aux = abs(X)
    Y_aux = abs(Y)
    compute_kernel += 1 / (1 - pow(euclidean_distances(X_aux, Y_aux, squared=True), d)) # General T-Student
    return compute_kernel


def linear_kernel(vector_x, vector_y):
    return np.dot(vector_x, vector_y)


def polynomial_kernel(vector_x, vector_y):
    power = 2
    return np.power(np.dot(vector_x, vector_y) + 1, power)


def gaussian_kernel(X, Y, r=None):
    if r is None:
        r = X.shape[1]
    K = euclidean_distances(X, Y, squared=True)*(-(r**-2))

    return np.exp(K)



# KernelFunction = linear_kernel
# KernelFunction = polynomial_kernel
# KernelFunction = radial_basis_kernel
# KernelFunction = gaussian_kernel
