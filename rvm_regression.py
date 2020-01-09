import numpy as np

# Constants definition
CONVERGENCE = 1e-9
SEED = 42
PRUNNING_THRESHOLD = 1e6
np.random.seed(SEED)

def initializeAlpha(N):
    # Initialization of alpha assuming uniform scale priors
    return np.array([np.full(N+1,1e-4),np.arange(0,N+1,1)])

def kernel(x_m, x_n, kernel_type):
    if kernel_type == "linear_spline":
        compute_kernel = 1 + x_m * x_n + x_m * x_n * min(x_m, x_n) - ((x_m + x_n) / 2) * pow(min(x_m, x_n), 2) + pow(min(x_m, x_n), 3) / 3
    '''elif kernel_type == "exponential":
        compute_kernel = '''
    return compute_kernel

def calculateBasisFunction(X, kernel_type):
    basis_mat = np.zeros((len(X), len(X) + 1))
    for i in range(basis_mat.shape[0]):
        for j in range(basis_mat.shape[1]):
            if j == 0:
                basis_mat[i,j] = 1
            else:    
                basis_mat[i,j] = kernel(X[i], X[j-1], kernel_type) # Starting in i-1 because mat is N+1 length 
    return basis_mat

def calculateA(alpha):
    return alpha * np.identity(len(alpha))

def calculateSigma(variance, Basis, A): # Already squared
    return np.linalg.inv(pow(variance, -1) * np.dot(np.transpose(Basis), Basis) + A)

def calculateMu(variance, Sigma, Basis, targets):
    return pow(variance, -1) * np.dot(np.dot(Sigma, np.transpose(Basis)), targets)

def updateHyperparameters(Sigma, alpha_old, mu, targets, Basis):
    # Update gammas
    
    gamma = np.zeros(len(alpha_old))
    for i in range(len(gamma)):
        gamma[i] = 1 - alpha_old[i] * Sigma[i,i]

    # Update alphas
    alpha = np.zeros(len(alpha_old))
    for i in range(len(alpha)):
        alpha[i] = gamma[i] / pow(mu[i], 2)

    # Update variance
    variance = pow(np.linalg.norm(targets - np.dot(Basis, mu)), 2) / (len(targets) - np.sum(gamma))

    return alpha, variance

def computeProbability(targets, variance, Basis, A):
    # Compute the Log Likelihood
    mat = variance * np.identity(len(targets)) + np.dot(np.dot(Basis, np.linalg.inv(A)), np.transpose(Basis))
    result = -1/2 * np.log(np.linalg.det(mat) + np.dot(np.transpose(targets), np.dot(np.linalg.inv(mat), targets)))
    return result

def prunning(alpha, Basis):
    index = []
    for i in range(len(alpha[0])):
        if (alpha[0][i] > PRUNNING_THRESHOLD):
            index.append(i)

    alpha = np.delete(alpha, index, 1)
    Basis = np.delete(Basis, index, 1)
    return alpha, Basis

def fit(X, variance, targets, kernel, N):
    previous_prob = float('inf')
    prob = 0
    alpha = initializeAlpha(N)
    Basis = calculateBasisFunction(X, kernel)
    A = calculateA(alpha[0])
    sigma = calculateSigma(variance, Basis, A)
    mu = calculateMu(variance, sigma, Basis, targets)
    cnt = 0
    while (True):
        alpha[0], variance = updateHyperparameters(sigma, alpha[0], mu, targets, Basis)
        alpha, Basis = prunning(alpha, Basis)
        A = calculateA(alpha[0])
        sigma = calculateSigma(variance, Basis, A)
        mu = calculateMu(variance, sigma, Basis, targets)
        prob = computeProbability(targets, variance, Basis, A)
        if (abs(prob - previous_prob) < CONVERGENCE): # Condition for convergence
            break
        previous_prob = prob
        cnt += 1
    print("Iterations:", cnt)
    return alpha, variance, mu, sigma

def predict(X_test, relevant_vectors, variance, mu, sigma, kernel):
    targets_predict = np.zeros(len(X_test))
    Basis = calculateBasisFunction(X_test, kernel)
    Basis = Basis[:,relevant_vectors]
    for i in range(len(X_test)):
        mean = np.dot(np.transpose(mu), Basis[i])
        var = variance + np.dot(np.dot(np.transpose(Basis[i]), sigma), Basis[i])
        targets_predict[i] = np.random.normal(mean, var)
    return targets_predict
