import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

import Kernel


# Formulas are taken from the paper
class RVM_Classifier():

    def __init__(self):
        self.threshold_alpha = 1e5
        self.bias_used = True
        self.removed_bias = False

        self.trained_weights = None
        self.relevance_vector = None

        self.training_data = None
        self.training_labels = None
        self.test_data = None
        self.test_labels = None

    def set_training_data(self, training_data, training_labels):
        self.training_data = training_data
        self.training_labels = training_labels

    def set_predefined_training_data(self, data_set, data_set_index=1):
        self.training_data = np.loadtxt(
            "datasets/{data_set}/{data_set}_train_data_{index}.asc".format(data_set=data_set, index=data_set_index))
        self.training_labels = np.loadtxt(
            "datasets/{data_set}/{data_set}_train_labels_{index}.asc".format(data_set=data_set, index=data_set_index))
        self.test_data = np.loadtxt(
            "datasets/{data_set}/{data_set}_test_data_{index}.asc".format(data_set=data_set, index=data_set_index))
        self.test_labels = np.loadtxt(
            "datasets/{data_set}/{data_set}_test_labels_{index}.asc".format(data_set=data_set, index=data_set_index))

    # Formula 27
    def update_weight_function(self, sigma, phi, beta, target):
        return np.linalg.multi_dot([sigma, phi.T, beta, target])

    # From formula after 16 before 18
    def gamma_function(self, alpha, sigma):
        gamma = np.zeros(len(alpha))
        for i in range(len(alpha)):
            gamma[i] = 1 - alpha[i] * sigma[i][i]
        return gamma

    # From formula 16
    def recalculate_alphas_function(self, alpha, gamma, weight):
        new_alphas = np.zeros(len(alpha))
        for i in range(len(gamma)):
            new_alphas[i] = gamma[i] / (weight[i] ** 2 + sys.float_info.epsilon)
        return new_alphas

    # def phi_function(self, x):
    #     phi = np.ones((x.shape[0], x.shape[0] + 1))
    #
    #     for n in range(x.shape[0]):
    #         for m in range(x.shape[0]):
    #             phi[n][m + 1] = Kernel.radial_basis_kernel(x[n, :], x[m, :])
    #     return phi

    def phi_function(self, x):
        phi = np.ones((x.shape[0], self.relevance_vector.shape[0] + 1))

        for n in range(x.shape[0]):
            for m in range(self.relevance_vector.shape[0]):
                phi[n][m + 1] = Kernel.radial_basis_kernel(x[n, :], self.relevance_vector[m, :])
        return phi

    # Formula 2
    def y_function(self, weight, x, phi):
        w0 = weight[0]  # TODO we dont use this anymore
        y = np.zeros(x.shape[0])
        # phi_sum = np.sum(phi,axis=1)
        # for n in range(x.shape[0]):
        # for m in range(n, x.shape[0]):
        #    y[n] += weight[n+1]*phi_sum[n]
        test = np.dot(phi, weight)
        y = expit(test)  # Sigmoid function

        return y

    # From under formula 25
    def beta_matrix_function(self, y, N):
        beta_matrix = np.zeros((N, N))
        for n in range(N):
            beta_matrix[n][n] = expit(y[n]) * (1 - expit(y[n]))
        # print(beta_matrix)
        return beta_matrix

    # Formula 26
    def sigma_function(self, phi, beta, alpha):
        b = np.linalg.multi_dot([phi.T, beta, phi])
        # print("b")
        # print(b)
        return np.linalg.inv(b + np.diag(alpha))

    # Formula 28 and
    def log_posterior_function(self, x, weight, target, phi, alpha):
        log_posterior = 0
        y = self.y_function(weight, x, phi)
        # print("min")
        # print(min(y))
        for n in range(x.shape[0]):
            # for k in range(target.shape[1]):
            # posterior += target[n] * np.log(sigmoid_function(y[n])) + (1-target[n])*np.log(1-sigmoid_function(y[n]))
            # print("aaaaaaaaaaaaaa")
            # print(y[n])

            log_posterior += target[n] * np.log(y[n])  # + (1-target[n])*np.log(1-y[n])
        log_posterior = log_posterior - np.linalg.multi_dot([weight.T, np.diag(alpha), weight])

        return log_posterior

    # Todo This function is alot of copy paste ceck it before doing more with it
    def prune(self, alphas, weights, phi, gamma, sigma):
        keep_alpha = alphas < self.threshold_alpha

        if not np.any(keep_alpha):
            keep_alpha[0] = True
            if self.bias_used:
                keep_alpha[-1] = True

        if self.bias_used:
            if not keep_alpha[-1]:
                self.bias_used = False
            self.relevance_vector = self.relevance_vector[keep_alpha[:-1]]
        else:
            self.relevance_vector = self.relevance_vector[keep_alpha]

        # if not keep_alpha[0] and not self.removed_bias:
        #     self.removed_bias = True


        return alphas[keep_alpha], weights[keep_alpha], phi[:, keep_alpha], gamma[keep_alpha], sigma[keep_alpha]

    # def plot(self, data, data_index_target_list):
    def plot(self, data=None, target=None):
        if data == None and target == None:
            data = self.test_data
            target = self.test_labels

        # Format data so we can plot it
        print("Will start plotting")
        target_values = np.unique(target)
        data_index_target_list = [0] * len(target_values)
        for i, c in enumerate(target_values):  # Saving the data index with its corresponding target
            data_index_target_list[i] = (c, np.argwhere(target == c))

        h = 0.1  # step size in the mesh
        # create a mesh to plot in
        x_min, x_max = data[:, 0].min(), data[:, 0].max()  # Gets the max and min value of x in the data
        y_min, y_max = data[:, 1].min(), data[:, 1].max()  # Gets the max and min value of y in the data
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))  # Creates a mesh from max and min with step size h

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        print("Calculating the prediction, this might take a while...")
        data_mesh = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(data_mesh)
        # Z = self.predict(data)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        colors = ["bo", "go", "ro", "co", "mo", "yo", "bo", "wo"]
        plt.figure(figsize=(12, 6))
        plt.title('Banana dataset')
        for i, c in enumerate(data_index_target_list):
            plt.plot(data[c[1], 0], data[c[1], 1], colors[i], label="Target: " + str(c[0]))
        plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
        # plt.scatter(self.relevance_vector[:, 0], self.relevance_vector[:, 1], c='black', marker='+', s=500)
        plt.xlabel("Cool label, what should we put here")
        plt.ylabel("Heelloooo lets goo")
        plt.legend()
        plt.show()

    def predict(self, data):
        """
            Runs the classification
        """
        phi = self.phi_function(data)
        y = self.y_function(self.trained_weights, data, phi)
        # y = np.array([expit(y) for y in estimations])
        pred = np.where(y > 0.5, 1, -1)
        return pred

    def train(self):
        """
         Train the classifier
        """
        self.relevance_vector = self.training_data
        max_training_iterations = 100
        threshold = 0.0000000000001

        phi = self.phi_function(self.training_data)
        # print("phi")
        # print(phi)

        old_log_posterior = float("-inf")
        weights = np.array(
            [1 / (self.training_data.shape[0] + 1)] * (self.training_data.shape[0] + 1))  # Initialize uniformly
        alpha = np.array([1 / (self.training_data.shape[0] + 1)] * (self.training_data.shape[0] + 1))
        for i in range(max_training_iterations):
            print("number of iterations:")
            print(i)

            y = self.y_function(weights, self.training_data, phi)
            # print("y")
            # print(y)

            beta = self.beta_matrix_function(y, self.training_data.shape[0])
            # print("beta")
            # print(beta)

            sigma = self.sigma_function(phi, beta, alpha)
            # print("sigma")
            # print(sigma)

            gamma = self.gamma_function(alpha, sigma)

            weights = self.update_weight_function(sigma, phi, beta, self.training_labels)

            # print("weight")
            # print(weights)

            alpha = self.recalculate_alphas_function(alpha, gamma, weights)

            alpha, weights, phi, gamma, sigma = self.prune(alpha, weights, phi, gamma, sigma)

            # print("alpha")
            # print(alpha)

            # print("y")
            # print(y)
            # print(min(y))
            log_posterior = self.log_posterior_function(self.training_data, weights, self.training_labels, phi, alpha)

            difference = log_posterior - old_log_posterior
            if difference <= threshold:
                print("Training done, it converged. Nr iterations: " + str(i + 1))
                self.trained_weights = weights
                break
            old_log_posterior = log_posterior

        # Format data so we can plot it
        target_values = np.unique(self.training_labels)
        data_index_target_list = [0] * len(target_values)
        for i, c in enumerate(target_values):  # Saving the data index with its corresponding target
            data_index_target_list[i] = (c, np.argwhere(self.training_labels == c))
