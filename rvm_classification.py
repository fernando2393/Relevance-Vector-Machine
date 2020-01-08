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


def phi_function(x):
    phi = np.ones((x.shape[0], x.shape[0] + 1))

    for m in range(x.shape[0]):
        for n in range(x.shape[0]):
            phi[m][n+1] = Kernel.radial_basis_kernel(x[n, :], x[m, :])
    return phi


# Before formula 23
def sigmoid_function(y):
    denominator = 1 + math.exp(-y)
    return 1 / denominator


# Formula 2
def y_function(weight, x):
    w0 = weight[0]
    y = np.zeros(x.shape[0])
    for n in range(x.shape[0]):
        num = 0
        for k in range(len(weight)):
            # for m in range(n, x.shape[0]):
            num += weight[k] * Kernel.radial_basis_kernel(x, x[n, :])
        y[n] = np.add(num, w0)
    return y


# From under formula 25
def beta_matrix_function(y, N, target):
    beta_matrix = np.zeros((N, N))
    for n in range(N):
        beta_matrix[n][n] = np.power(sigmoid_function(y[n]), target[n]) * np.power((1 - sigmoid_function(y[n])), (1-target[n]))
    #print(beta_matrix)
    return beta_matrix


# Formula 26
def sigma_function(phi, beta, alpha):
    print("phi")
    print(phi)
    print("beta")
    print(beta)
    b = np.dot(phi.T, np.dot(beta, phi))
    print("b")
    print(b)
    # print("alpha")
    # print(alpha)
    final = b + alpha

    return np.linalg.inv(final)


# Formula 27
def update_weight_function(sigma, phi, beta, target):
    print("sigma")
    print(sigma)
    # print("phi")
    # print(phi.T)
    # print("beta")
    # print(beta)
    # print("target")
    # print(target)
    return np.dot(sigma, np.dot(phi.T , np.dot(beta , target)))


# Formula 28 and
def log_posterior_function(x, weight, target):
    posterior = 0
    y = y_function(weight, x)
    for n in range(x.shape[0]):
        #for k in range(target.shape[1]):
        posterior += target[n] * np.log(sigmoid_function(y[n]))
    return posterior


def run():
    """
        Runs the classification
    """


def train(data, target):
    """
     Train the classifier
    """
    max_training_iterations = 10000
    threshold = 0.0001

    old_log_posterior = float("-inf")
    weights = np.array([1 / (data.shape[0]+1)] * (data.shape[0]+1))  # Initialize uniformly
    alpha = np.array([1 / (data.shape[0]+1)] * (data.shape[0]+1))
    for i in range(max_training_iterations):
        print("1")
        y = y_function(weights, data)
        print(y.shape)

        print(data.shape)

        beta = beta_matrix_function(y, data.shape[0], target)
        print(beta.shape)

        phi = phi_function(data)
        print(phi.shape)
        
        sigma = sigma_function(phi, beta, alpha)
        print(sigma.shape)

        gamma = gamma_function(alpha, sigma)
        print(gamma.shape)

        mu = mu_function(beta, sigma, phi, target)
        print(mu.shape)

        alpha = recalculate_alphas_function(alpha, gamma, mu)
        print(weights.shape)

        weights = update_weight_function(sigma, phi, beta, target)
        print(weights.shape)

        log_posterior = log_posterior_function(data, weights, target)

        difference = log_posterior - old_log_posterior
        if difference <= threshold:
            print("Training done, it converged")
            break
        old_log_posterior = log_posterior
