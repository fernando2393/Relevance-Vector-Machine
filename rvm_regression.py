import numpy as np

# Constants definition
CONVERGENCE = 1e-9
PRUNNING_THRESHOLD = 1e3

def initializeAlpha(N):
    # Initialization of alpha assuming uniform scale priors
    return np.array([np.full(N+1,1e-4),np.arange(0,N+1,1)])

def kernel(x_m, x_n, kernel_type):
    if (kernel_type == "linear_spline"):
        compute_kernel = 1 + x_m * x_n + x_m * x_n * np.minimum(x_m, x_n) - ((x_m + x_n) / 2) * pow(np.minimum(x_m, x_n), 2) + pow(np.minimum(x_m, x_n), 3) / 3
    elif (kernel_type == "rbf"):
        compute_kernel = rbf_kernel(x_m, x_n, 0.01)
        
    return compute_kernel.mean()

def calculateBasisFunction(X, kernel_type):
    Basis = np.zeros((X.shape[0], X.shape[0]))
    for i in range(Basis.shape[0]):
        for j in range(Basis.shape[1]):
            Basis[i,j] = kernel(X[i], X[j], kernel_type)

    weight_0 = np.ones((X.shape[0],1))
    Basis = np.hstack((weight_0, Basis))

    return Basis

def calculateA(alpha):
    return alpha * np.identity(len(alpha))

def calculateSigma(variance, Basis, A): # Already squared
    return np.linalg.inv(pow(variance, -1) * np.dot(np.transpose(Basis), Basis) + A)

def calculateMu(variance, Sigma, Basis, targets, N):
    return pow(variance, -1) * np.dot(np.dot(Sigma, np.transpose(Basis)), targets)

def updateHyperparameters(Sigma, alpha_old, mu, targets, Basis, N):
    # Update gammas
    
    gamma = np.zeros(len(alpha_old))
    for i in range(len(gamma)):
        gamma[i] = 1 - alpha_old[i] * Sigma[i,i]

    # Update alphas
    alpha = np.zeros(len(alpha_old))
    for i in range(len(alpha)):
        alpha[i] = gamma[i] / pow(mu[i], 2)

    # Update variance
    variance = pow(np.linalg.norm(targets - np.dot(Basis, mu)), 2) / (N - np.sum(gamma))

    return alpha, variance

def computeProbability(targets, variance, Basis, A):
    # Compute the Log Likelihood
    mat = variance * np.identity(len(targets)) + np.dot(np.dot(Basis, np.linalg.inv(A + np.eye(A.shape[1])*1e-27)), np.transpose(Basis))
    result = -1/2 * np.log(np.linalg.det(mat + np.eye(mat.shape[1])*1e-27) + np.dot(np.transpose(targets), np.dot(np.linalg.inv(mat + np.eye(mat.shape[1])*1e-27), targets)))
    return result

def prunning(alpha, Basis):
    index = []
    for i in range(len(alpha[0])):
        if (i != 0 and alpha[0][i] > PRUNNING_THRESHOLD):
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
    mu = calculateMu(variance, sigma, Basis, targets, N)
    cnt = 0
    while (True):
        alpha[0], variance = updateHyperparameters(sigma, alpha[0], mu, targets, Basis, N)
        alpha, Basis = prunning(alpha, Basis)
        A = calculateA(alpha[0])
        sigma = calculateSigma(variance, Basis, A)
        mu = calculateMu(variance, sigma, Basis, targets, N)
        prob = computeProbability(targets, variance, Basis, A)
        if (abs(prob - previous_prob) < CONVERGENCE): # Condition for convergence
            break
        previous_prob = prob
        cnt += 1
    print("Iterations:", cnt)
    return alpha, variance, mu, sigma

def predict(X_train, X_test, relevant_vectors, variance, mu, sigma, kernel_type, dimensions):
    targets_predict = np.zeros(len(X_test))
    X_samples = np.zeros((len(relevant_vectors)-1, dimensions))
    
    for i in range(1,len(relevant_vectors)):
        X_samples[i-1] = X_train[relevant_vectors[i]-1]
    
    Basis = np.zeros((len(X_test), X_samples.shape[0]+1))
    for i in range(len(X_test)):
        for j in range(len(X_samples)+1):
            if (j == 0):
                Basis[i,j] = 1
            else:
                Basis[i,j] = kernel(X_test[i], X_samples[j-1], kernel_type)

    for i in range(len(X_test)):
        mean = np.dot(np.transpose(mu), Basis[i])
        var = variance + np.dot(np.dot(np.transpose(Basis[i]), sigma), Basis[i])
        targets_predict[i] = np.random.normal(mean, var)
    return targets_predict
