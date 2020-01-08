import math

import numpy as np


# All defined kernels are defined in this class
def linear_kernel(vector_x, vector_y):
    return np.dot(vector_x, vector_y)


def polynomial_kernel(vector_x, vector_y):
    power = 2
    return np.power(np.dot(vector_x, vector_y) + 1, power)


def radial_basis_kernel(vector_x, vector_y):
    sigma = 2
    return math.exp(-math.pow(np.linalg.norm(np.subtract(vector_x, vector_y)), 2) / (2 * math.pow(sigma, 2)))


# KernelFunction = linear_kernel
# KernelFunction = polynomial_kernel
# KernelFunction = radial_basis_kernel
# KernelFunction = gaussian_kernel
