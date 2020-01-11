import numpy as np
from tqdm import tqdm

# Constants definition
CONVERGENCE = 1e-3
PRUNNING_THRESHOLD = 1e9

def initializeAlpha(N):
    # Initialization of alpha assuming uniform scale priors
    return np.array([np.full(N+1,1e-4),np.arange(0,N+1,1)])

def kernel(x_m, x_n, kernel_type):
    if (kernel_type == "linear_spline"):
        xmin = np.minimum(x_m, x_n)
        compute_kernel = 1 + x_m * x_n + x_m * x_n * xmin - ((x_m + x_n) / 2) * pow(xmin, 2) + pow(xmin, 3) / 3
    elif (kernel_type == "exponential"):
        eta_1 = 997*1e-4 
        eta_2 = 2*1e-4
        xmin_1 = np.minimum(x_m[0], x_n[0])
        xmin_2 = np.minimum(x_m[1], x_n[1])
        compute_kernel = np.exp(- eta_1*pow(xmin_1, 2) - eta_2*pow(xmin_2, 2))
    else:
        print("Please, select an suitable kernel")
    return compute_kernel.prod()

def calculateBasisFunction(X, kernel_type):
    Basis = np.zeros((X.shape[0], X.shape[0]))
    for i in tqdm(range(Basis.shape[0])):
        for j in range(Basis.shape[1]):
            Basis[i,j] = kernel(X[i], X[j], kernel_type)

    weight_0 = np.ones((X.shape[0],1))
    Basis = np.hstack((weight_0, Basis))

    return Basis

def calculateA(alpha):
    return alpha * np.identity(len(alpha))

def calculateSigma(variance, Basis, A): # Already squared
    return np.linalg.inv(pow(variance, -1) * np.dot(np.transpose(Basis), Basis) + A + np.eye(A.shape[0])*1e-9)

def calculateMu(variance, Sigma, Basis, targets, N):
    return pow(variance, -1) * np.dot(np.dot(Sigma, np.transpose(Basis)), targets)

def updateHyperparameters(Sigma, alpha, mu, targets, Basis, N):
    # Update gammas
    
    gamma = np.zeros(len(alpha))
    for i in range(len(gamma)):
        gamma[i] = 1 - alpha[i] * Sigma[i,i]

    # Update alphas
    alpha = np.zeros(len(alpha))
    for i in range(len(alpha)):
        alpha[i] = gamma[i] / pow(mu[i], 2)

    # Update variance
    variance = pow(np.linalg.norm(targets - np.dot(Basis, mu)), 2) / (N - np.sum(gamma))

    return alpha, variance

def computeLogLikelihood(targets, variance, Basis, A, N):
    '''
    In some scenarios this convergency criteria doesn't
    satisfy.Therefore we converged based on alphas
    '''
    # Compute the Log Likelihood
    posterior_weight_cov = np.linalg.inv(A + np.dot(variance, np.dot(np.transpose(Basis), Basis)))
    posterior_weight_mean = np.dot(variance, np.dot(posterior_weight_cov, np.dot(np.transpose(Basis), targets)))
    first_term = - np.log(np.linalg.det(posterior_weight_cov)) - N * np.log(variance) - np.log(np.linalg.det(A))
    second_term = np.dot(variance, pow(np.linalg.norm(targets - np.dot(Basis, posterior_weight_mean)), 2)) + np.dot(np.transpose(posterior_weight_mean), np.dot(A, posterior_weight_mean))
    result = -1/2 * (first_term + second_term)
    return result

def prunning(alpha, Basis, alpha_old):
    index = []
    for i in range(len(alpha[0])):
        if (i != 0 and alpha[0][i] > PRUNNING_THRESHOLD):
            index.append(i)

    alpha = np.delete(alpha, index, 1)
    Basis = np.delete(Basis, index, 1)
    alpha_old = np.delete(alpha_old, index, 0)
    return alpha, Basis, alpha_old

def fit(X, variance, targets, kernel, N):
    alpha = initializeAlpha(N)
    Basis = calculateBasisFunction(X, kernel)
    A = calculateA(alpha[0])
    sigma = calculateSigma(variance, Basis, A)
    mu = calculateMu(variance, sigma, Basis, targets, N)
    cnt = 0
    alpha_old = alpha[0].copy()
    while (True and cnt < 1000):
        alpha[0], variance = updateHyperparameters(sigma, alpha[0], mu, targets, Basis, N)
        alpha, Basis, alpha_old = prunning(alpha, Basis, alpha_old)
        A = calculateA(alpha[0])
        sigma = calculateSigma(variance, Basis, A)
        mu = calculateMu(variance, sigma, Basis, targets, N)
        # prob = computeLogLikelihood(targets, variance, Basis, A, N)
        diff = np.absolute(max(alpha[0]) - max(alpha_old))
        if (diff < CONVERGENCE and cnt > 100): # Condition for convergence
            break
        if (cnt%1000 == 0 and cnt != 0):
            print('Difference:', diff)
        alpha_old = alpha[0].copy()
        cnt += 1
    print("Iterations:", cnt)
    return alpha, variance, mu, sigma

def predict(X_train, X_test, relevant_vectors, variance, mu, sigma, kernel_type, dimensions):
    targets_predict = np.zeros(len(X_test))
    X_samples = np.zeros((len(relevant_vectors)-1, dimensions))
    
    for i in range(1,len(relevant_vectors)):
        X_samples[i-1] = X_train[relevant_vectors[i]-1]
    
    Basis = np.zeros((len(X_test), X_samples.shape[0]+1))

    for i in tqdm(range(len(X_test))):
        for j in range(len(X_samples)+1):
            if (j == 0):
                Basis[i,j] = 1
            else:
                Basis[i,j] = kernel(X_test[i], X_samples[j-1], kernel_type)
    
    for i in tqdm(range(len(X_test))):
        targets_predict[i] = np.dot(np.transpose(mu), Basis[i])
    return targets_predict