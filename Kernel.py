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


# From 4.2
def gaussian_kernel(vector_x, vector_y):
    r = 0.5  # Width of the     parameter, 0.5 was used in the paper
    return math.exp(-(r ** -2) * np.linalg.norm(vector_x, vector_y.T))


# KernelFunction = linear_kernel
# KernelFunction = polynomial_kernel
# KernelFunction = radial_basis_kernel
KernelFunction = gaussian_kernel
