import math
import sys

import matplotlib.pyplot as plt
import numpy as np

import Kernel


# Formulas are taken from the paper
class RVM_Classifier():

    def __init__(self):
        self.trained_weights = None

    # From formula 27
    def mu_function(self, betas, sigma, phi, targets):
        return np.dot(sigma, np.dot(phi.T, np.dot(betas, targets)))

    # From formula after 16 before 18
    def gamma_function(self, alpha, sigma):
        gamma = np.zeros(len(alpha))
        for i in range(len(alpha)):
            gamma[i] = 1 - alpha[i] * sigma[i][i]
        return gamma

    # From formula 16
    def recalculate_alphas_function(self, alpha, gamma, mu):
        new_alphas = np.zeros(len(alpha))
        for i in range(len(gamma)):
            new_alphas[i] = gamma[i] / ((mu[i] ** 2) + sys.float_info.epsilon)
        return new_alphas

    def phi_function(self, x):
        phi = np.ones((x.shape[0], x.shape[0] + 1))

        for m in range(x.shape[0]):
            for n in range(x.shape[0]):
                phi[m][n + 1] = Kernel.radial_basis_kernel(x[n, :], x[m, :])
        return phi

    # Before formula 23   # Todo This function exists in scipy and might be more optimized there. (scipy.special.expit)
    def sigmoid_function(self, y):
        denominator = 1 + math.exp(-y)
        return 1 / denominator

    # Formula 2
    def y_function(self, weight, x):
        w0 = weight[0]
        y = np.zeros(x.shape[0])
        for n in range(x.shape[0]):
            num = 0
            for k in range(len(weight)):
                num += weight[k] * Kernel.radial_basis_kernel(x, x[n, :])
            y[n] = np.add(num, w0)
        return y

    # From under formula 25
    def beta_matrix_function(self, y, N, target):
        beta_matrix = np.zeros((N, N))
        for n in range(N):
            beta_matrix[n][n] = np.power(self.sigmoid_function(y[n]), target[n]) * np.power(
                (1 - self.sigmoid_function(y[n])), (1 - target[n]))
        # print(beta_matrix)
        return beta_matrix

    # Formula 26
    def sigma_function(self, phi, beta, alpha):
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
    def update_weight_function(self, sigma, phi, beta, target):
        return np.dot(sigma, np.dot(phi.T, np.dot(beta, target)))

    # Formula 28 and
    def log_posterior_function(self, x, weight, target):
        posterior = 0
        y = self.y_function(weight, x)
        for n in range(x.shape[0]):
            posterior += target[n] * np.log(self.sigmoid_function(y[n]))
        return posterior

    # def plot(self, data, data_index_target_list):
    def plot(self, data, target):
        # Format data so we can plot it
        print("Will start plotting fucntion")
        target_values = np.unique(target)
        data_index_target_list = [0] * len(target_values)
        for i, c in enumerate(target_values):  # Saving the data index with its corresponding target
            data_index_target_list[i] = (c, np.argwhere(target == c))

        h = 0.1  # step size in the mesh
        # create a mesh to plot in
        x_min, x_max = data[:, 0].min(), data[:, 0].max()   # Gets the max and min value of x in the data
        y_min, y_max = data[:, 1].min(), data[:, 1].max()   # Gets the max and min value of y in the data
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))  # Creates a mesh from max and min with step size h

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        data_mesh = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(data_mesh)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)



        colors = ["bo", "go", "ro", "co", "mo", "yo", "bo", "wo"]
        # plt.figure(figsize=(10, 6))
        plt.title('Banana dataset')
        for i, c in enumerate(data_index_target_list):
            plt.plot(data[c[1], 0], data[c[1], 1], colors[i], label="Target: " + str(c[0]))
        plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
        plt.xlabel("Cool label, what should we put here")
        plt.ylabel("Heelloooo lets goo")
        plt.legend()
        plt.show()

    def predict(self, data):
        """
            Runs the classification
        """
        estimations = self.y_function(self.trained_weights, data)     # Todo I limit this to 100 under dev
        y = np.array([self.sigmoid_function(y) for y in estimations])
        pred = np.where(y > 0.5, 1, -1)             # Todo only works when using 2 classes, also copy paste code
        return pred

    def train(self, data, target):
        """
         Train the classifier
        """
        max_training_iterations = 10000
        threshold = 0.0001

        old_log_posterior = float("-inf")
        weights = np.array([1 / (data.shape[0] + 1)] * (data.shape[0] + 1))  # Initialize uniformly
        alpha = np.array([1 / (data.shape[0] + 1)] * (data.shape[0] + 1))
        for i in range(max_training_iterations):
            y = self.y_function(weights, data)
            print(y.shape)

            print(data.shape)

            beta = self.beta_matrix_function(y, data.shape[0], target)
            print(beta.shape)

            phi = self.phi_function(data)
            print(phi.shape)

            sigma = self.sigma_function(phi, beta, alpha)
            print(sigma.shape)

            gamma = self.gamma_function(alpha, sigma)
            print(gamma.shape)

            mu = self.mu_function(beta, sigma, phi, target)
            print(mu.shape)

            alpha = self.recalculate_alphas_function(alpha, gamma, mu)
            print(weights.shape)

            weights = self.update_weight_function(sigma, phi, beta, target)
            print(weights.shape)

            log_posterior = self.log_posterior_function(data, weights, target)

            difference = log_posterior - old_log_posterior
            if difference <= threshold:
                print("Training done, it converged. Nr iterations: " + str(i + 1))
                self.trained_weights = weights
                break
            old_log_posterior = log_posterior

        # Format data so we can plot it
        target_values = np.unique(target)
        data_index_target_list = [0] * len(target_values)
        for i, c in enumerate(target_values):  # Saving the data index with its corresponding target
            data_index_target_list[i] = (c, np.argwhere(target == c))
        # self.plot(data, data_index_target_list)
