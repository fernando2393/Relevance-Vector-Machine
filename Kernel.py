import math

import numpy as np


# All defined kernels are defined in this class
from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances


def linear_kernel(vector_x, vector_y):
    return np.dot(vector_x, vector_y)


def polynomial_kernel(vector_x, vector_y):
    power = 2
    return np.power(np.dot(vector_x, vector_y) + 1, power)


def radial_basis_kernel(X, Y, gamma=0.5):
    # sigma = 2
    # return math.exp(-math.pow(np.linalg.norm(np.subtract(vector_x, vector_y)), 2) / (2 * math.pow(sigma, 2)))
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = euclidean_distances(X, Y, squared=True)
    K *= -gamma
    np.exp(K, K)  # exponentiate K in-place
    return K


# KernelFunction = linear_kernel
# KernelFunction = polynomial_kernel
# KernelFunction = radial_basis_kernel
# KernelFunction = gaussian_kernel
