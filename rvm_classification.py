import math

import numpy as np
from scipy.special import  expit

import Kernel


# Formulas are taken from the paper

# Formula 27
def update_weight_function(sigma, phi, beta, target):
    return np.linalg.multi_dot([sigma, phi.T , beta , target])


# From formula after 16 before 18
def gamma_function(alpha, sigma):
    gamma = np.zeros(len(alpha))
    for i in range(len(alpha)):
        gamma[i] = 1 - alpha[i] * sigma[i][i]
    return gamma


# From formula 16
def recalculate_alphas_function(alpha, gamma, weight):
    new_alphas = np.zeros(len(alpha))
    for i in range(len(gamma)):
        new_alphas[i] = gamma[i] / (weight[i] ** 2)
    return new_alphas


def phi_function(x):
    phi = np.ones((x.shape[0], x.shape[0] + 1))
    for n in range(x.shape[0]):
        for m in range(x.shape[0]):
            phi[n][m+1] = Kernel.radial_basis_kernel(x[n, :], x[m, :])
    return phi


# Before formula 23
def sigmoid_function(y):
    denominator = (1 + math.exp(y))
    return 1 / denominator


# Formula 2
def y_function(weight, x, phi):
    w0 = weight[0]
    y = np.zeros(x.shape[0])
    #phi_sum = np.sum(phi,axis=1)
    #for n in range(x.shape[0]):
        # for m in range(n, x.shape[0]):
    #    y[n] += weight[n+1]*phi_sum[n]
    test = np.dot(phi, weight)
    y = expit(test)
    
    return y


# From under formula 25
def beta_matrix_function(y, N, target):
    beta_matrix = np.zeros((N, N))
    for n in range(N):
        beta_matrix[n][n] = sigmoid_function(y[n]) * (1 - sigmoid_function(y[n]))
    #print(beta_matrix)
    return beta_matrix


# Formula 26
def sigma_function(phi, beta, alpha):
    b = np.linalg.multi_dot([phi.T, beta, phi])
    # print("b")
    # print(b)
    return np.linalg.inv(b+np.diag(alpha))


# Formula 28 and
def log_posterior_function(x, weight, target,phi, alpha):
    log_posterior = 0
    y = y_function(weight, x, phi)
    # print("min")
    # print(min(y))
    for n in range(x.shape[0]):
        #for k in range(target.shape[1]):
        #posterior += target[n] * np.log(sigmoid_function(y[n])) + (1-target[n])*np.log(1-sigmoid_function(y[n]))
        #print("aaaaaaaaaaaaaa")
        #print(y[n])
        
        log_posterior += target[n] * np.log(y[n]) #+ (1-target[n])*np.log(1-y[n])
    log_posterior = log_posterior - np.linalg.multi_dot([weight.T, np.diag(alpha), weight])    
     
    return log_posterior


def prune(self):
    mask = self.alphas < self.threshold_alpha

    self.alphas = self.alphas[mask]
    self.old_alphas = self.old_alphas[mask]
    self.phi = self.phi[:, mask]
    self.mu_posterior = self.mu_posterior[mask]
    
    if not self.removed_bias:
        self.relevance_vec = self.relevance_vec[mask[1:]]
    else:
        self.relevance_vec = self.relevance_vec[mask]
        
    if not mask[0] and not self.removed_bias:
        self.removed_bias = True
        if self.verbose:
            print("Bias removed")
        


def run():
    """
        Runs the classification
    """


def train(data, target):
    """
     Train the classifier
    """
    max_training_iterations = 100
    threshold = 0.0000000000001

    old_log_posterior = float("-inf")
    weights = np.array([1 / (data.shape[0]+1)] * (data.shape[0]+1))  # Initialize uniformly
    alpha = np.array([1 / (data.shape[0]+1)] * (data.shape[0]+1))
    for i in range(max_training_iterations):
        print("number of iterations:")
        print(i)
        
        phi = phi_function(data)
        # print("phi")
        # print(phi)

        y = y_function(weights, data,phi)       
        # print("y")
        # print(y)
        
        beta = beta_matrix_function(y, data.shape[0], target)
        # print("beta")
        # print(beta)
        
        sigma = sigma_function(phi, beta, alpha)
        # print("sigma")
        # print(sigma)
        
        gamma = gamma_function(alpha, sigma)
        
        weights = update_weight_function(sigma, phi, beta, target)
        
        # print("weight")
        # print(weights)

        alpha = recalculate_alphas_function(alpha, gamma, weights)
        # print("alpha")
        # print(alpha)        
        
        # print("y")
        # print(y)
        # print(min(y))
        log_posterior = log_posterior_function(data, weights, target, phi, alpha)

        difference = log_posterior - old_log_posterior
        if difference <= threshold:
            print("Training done, it converged")
            break
        old_log_posterior = log_posterior
