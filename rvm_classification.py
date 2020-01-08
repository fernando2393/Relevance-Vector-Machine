import math

import numpy as np

import Kernel


# Formulas are taken from the paper

# From formula 27
def mu_function(betas, sigma, phi, targets):
    return np.dot(sigma, np.dot(phi.T, np.dot(betas, targets)))


# From formula after 16 before 18
def gamma_function(alpha, sigma):
    gamma = np.zeros(len(alpha))
    for i in range(len(alpha)):
        gamma[i] = 1 - alpha[i] * sigma[i][i]
    return gamma


# From formula 16
def recalculate_alphas_function(alpha, gamma, mu):
    new_alphas = np.zeros(len(alpha))
    for i in range(len(gamma)):
        new_alphas[i] = gamma[i] / (mu[i] ** 2)
    return new_alphas


# Before formula 23
def sigmoid_function(y):
    denominator = 1 + math.exp(-y)
    return 1 / denominator


# From under formula 25
def beta_matrix_function(y):
    beta_matrix = np.zeros((len(y), len(y)))
    for i in range(len(y)):
        beta_matrix[i][i] = sigmoid_function(y[i]) * (1 - sigmoid_function(y[i]))
    return beta_matrix


# Formula 26
def sigma_function(phi, beta, alpha):
    return np.linalg.inv(phi.T * beta * phi + alpha)


# Formula 27
def update_weight_function(sigma, phi, beta, target):
    return sigma * phi.T * beta * target


# Formula 28 and
def log_posterior_function(x, weight, target):
    posterior = 0
    for n in range(x.shape[0]):
        for k in range(len(weight)):
            y = y_function(weight, x)
            posterior += target[n][k] * np.log(sigmoid_function(y[n][k]))
    return posterior


# Formula 2
def y_function(weight, x):
    w0 = weight[0]
    y = np.zeros((x.shape[0], len(weight)))
    for k in range(len(weight)):
        num = 0
        for n in range(x.shape[0]):
            for n2 in range(n, x.shape[0]):
                num += weight * Kernel.gaussian_kernel(x[n, :], x[n2, :])
            y[n][k] = num + w0
    return y


def run():
    print("hi")