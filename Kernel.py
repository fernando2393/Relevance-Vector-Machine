import math

import numpy as np


# All defined kernels are defined in this class
from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances


def linear_kernel(vector_x, vector_y):
    return np.dot(vector_x, vector_y)


def polynomial_kernel(vector_x, vector_y):
    power = 2
    return np.power(np.dot(vector_x, vector_y) + 1, power)


def radial_basis_kernel(X, Y, r=0.5):
    #X, Y = check_pairwise_arrays(X, Y)
    distance = euclidean_distances(X, Y, squared=True)
    kernel = -r**2 * distance
    return np.exp(kernel)


# KernelFunction = linear_kernel
# KernelFunction = polynomial_kernel
# KernelFunction = radial_basis_kernel
# KernelFunction = gaussian_kernel
