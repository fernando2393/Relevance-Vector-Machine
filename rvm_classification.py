import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

import Kernel


class RVM_Classifier:
    """
        Relevance vector machine classifier
    """

    def __init__(self):

        self.threshold_alpha = 1e11

        # If the bias is pruned we set this to True
        self.removed_bias = False

        self.alphas = None
        self.alphas_old = None
        self.phi = None
        self.weight = None
        self.relevance_vector = None

        # Training and test data
        self.training_data = None
        self.training_labels = None
        self.test_data = None
        self.test_labels = None
        self.n_classes = None

        # The prediction is stored here
        self.prediction = None

    def set_training_data(self, training_data, training_labels):
        self.training_data = training_data
        self.training_labels = training_labels
        self.training_labels[self.training_labels == -1] = 0  # Sanitize labels, some use -1 so we force it to 0

    def set_predefined_training_data(self, data_set, data_set_index=1, nr_samples=None):
        if data_set == "pima":
            data_training = pd.read_csv("datasets/pima/pima-training.csv", delim_whitespace=True)
            data_training['type'] = data_training['type'].map({'Yes': 1, 'No': 0})  # Translating to boolean
            data = data_training.values
            self.training_data = data[:, :-1]
            self.training_labels = data[:, -1]

            data_test = pd.read_csv("datasets/pima/pima-test.csv", delim_whitespace=True)
            data_test['type'] = data_test['type'].map({'Yes': 1, 'No': 0})  # Translating to boolean
            data = data_test.values
            self.test_data = data[:, :-1]
            self.test_labels = data[:, -1]
        elif data_set == "usps":
            raise NameError("This data set is not yet supported")
        elif data_set == "ripley":
            self.training_data = np.loadtxt("datasets/ripley/ripley_train_data.asc")
            self.training_labels = np.loadtxt("datasets/ripley/ripley_train_labels.asc")
            self.test_data = np.loadtxt("datasets/ripley/ripley_test_data.asc")
            self.test_labels = np.loadtxt("datasets/ripley/ripley_test_labels.asc")
        else:
            self.training_data = np.loadtxt(
                "datasets/{data_set}/{data_set}_train_data_{index}.asc".format(data_set=data_set, index=data_set_index))
            self.training_labels = np.loadtxt(
                "datasets/{data_set}/{data_set}_train_labels_{index}.asc".format(data_set=data_set,
                                                                                 index=data_set_index))
            self.training_labels[self.training_labels == -1] = 0  # Sanitize labels, some use -1 so we force it to 0

            self.test_data = np.loadtxt(
                "datasets/{data_set}/{data_set}_test_data_{index}.asc".format(data_set=data_set, index=data_set_index))
            self.test_labels = np.loadtxt(
                "datasets/{data_set}/{data_set}_test_labels_{index}.asc".format(data_set=data_set,
                                                                                index=data_set_index))
            self.test_labels[self.test_labels == -1] = 0  # Sanitize labels, some use -1 so we force it to 0

        if nr_samples is not None:
            random_training_data, random_training_target = self.get_nr_random_samples(self.training_data,
                                                                                      self.training_labels, nr_samples)
            self.training_data = random_training_data
            self.training_labels = random_training_target

            random_test_data, random_test_target = self.get_nr_random_samples(self.test_data, self.test_labels,
                                                                              nr_samples)
            self.test_data = random_test_data
            self.test_labels = random_test_target

    def multi_class_transformation(self, labels):
        labels = pd.get_dummies(labels).values
        return labels

    def get_nr_random_samples(self, data, target, nr_samples):
        total_nr_samples = data.shape[0]
        rnd_indexes = random.sample(range(total_nr_samples), nr_samples)

        random_data = []
        random_target = []
        for index in rnd_indexes:
            random_data.append(data[index])
            random_target.append(target[index])

        random_data = np.array(random_data)
        random_target = np.array(random_target)
        return random_data, random_target

    def get_nr_relevance_vectors(self):
        return self.relevance_vector.shape[0]

    # From formula 16
    def recalculate_alphas_function(self, gamma, weights):
        return gamma / (weights ** 2)

    # From formula after 16 before 18 (17)
    def gamma_function(self, alpha, sigma):
        return 1 - alpha * np.diag(sigma)

    # Formula 26. With alpha from below 13
    def sigma_function(self, phi, beta, alpha):
        """for banana data 1 using:
           1e-4 gets over 300 relevance vector,
           1e-3 we get 13
           1e-2 we get 9
           1e-1 we get 244
           This values might change for each data set, do not know what to do :(
       for me it makes sense to use 1e-3 because we do not want value that big that might interfere in the sigma computation, and to small is shit.
       another option is to reduce the threshold and use 1e11 instead of 1e12or even 1e9"""
        hessian = np.linalg.multi_dot([phi.T, beta, phi]) + np.diag(alpha)
        return np.linalg.inv(hessian)  # + np.eye(len(alpha))*1e-3)

    # From under formula 25
    def beta_matrix_function(self, y):
        return np.diag(y * (1 - y))

    # From under formula 4
    def phi_function(self, x, y):
        phi_kernel = Kernel.gaussian_kernel(x, y, r=None)
        """ there is a difference when estimating for the ripleys and the other data sets. 
        For kernel, in the paper, it is said that we should use r^2, where r=0.5. However, another interpretation is that r^2 = 0.5, in this case, our input must be sqrt(0.5).
        Seems a little bit off for me :("""
        if self.removed_bias:
            return phi_kernel
        phi0 = np.ones((phi_kernel.shape[0], 1))
        return np.hstack((phi0, phi_kernel))

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))

    # Formula 1
    def y_function(self, weight, phi):
        y = self.sigmoid(np.dot(phi, weight))
        return y

    # Formula 24
    def log_posterior_function(self, weight, alpha, phi, target):
        log_posterior = 0
        jacobian = []
        y = self.y_function(weight, phi)
        for k in range(self.n_classes):
            y_1 = y[target[:,k] == 1]
            sum_y = np.sum(np.log(y_1))
            log_posterior += sum_y - (np.linalg.multi_dot([weight.T, np.diag(alpha), weight]) / 2)
            jacobian.append(np.dot(np.diag(alpha), weight) - np.dot(phi.T, (target[:,k] - sum_y)))
        jacobian = np.sum(jacobian, axis=0)
        return -log_posterior, jacobian

    def hessian(self, weights, alphas, phi, target):
        y = self.y_function(weights, phi)
        beta = self.beta_matrix_function(y)
        hessian = np.zeros((np.diag(alphas).shape[0], np.diag(alphas).shape[0]))
        for k in range(self.n_classes):
            hessian += np.diag(alphas) + np.linalg.multi_dot([phi.T, beta, phi])
        return hessian

    def update_weights(self):
        result = minimize(
            fun=self.log_posterior_function,
            hess=self.hessian,
            x0=self.weight,
            args=(self.alphas, self.phi, self.training_labels),
            method='Newton-CG',
            jac=True,
            options={
                'maxiter': 50
            }
        )
        self.weight = result.x  # Updates the weights to the maximized (log is negative that is why we minimize)
        ok = 3
    def get_pruning_info(self):
        alphas_checked = (self.alphas < self.threshold_alpha).tolist()
        nr_to_prune = alphas_checked.count(False)

        index_pos = 0
        indexes = []
        for i in range(nr_to_prune):
            pos = alphas_checked.index(False, index_pos)
            indexes.append(pos)
            index_pos = pos + 1

        bias_removed = False
        if not alphas_checked[0]:
            bias_removed = True
        return indexes, bias_removed, alphas_checked

    def prune(self):
        indexes, bias_removed, alphas_checked = self.get_pruning_info()
        self.alphas = np.delete(self.alphas, indexes)
        self.alphas_old = np.delete(self.alphas_old, indexes)
        self.phi = np.delete(self.phi, indexes, 1)
        self.weight = np.delete(self.weight, indexes)

        if not self.removed_bias:
            bias_check = (np.array(indexes) - 1).tolist()  # Produces annoying warning
            self.relevance_vector = np.delete(self.relevance_vector, bias_check, 0)
        else:
            self.relevance_vector = np.delete(self.relevance_vector, indexes, 0)

        if bias_removed and not self.removed_bias:
            self.removed_bias = True
            print("Bias removed")

    def fit(self):
        """
            Train the classifier
        """
        self.training_labels = self.multi_class_transformation(self.training_labels)
        self.n_classes = self.training_labels.shape[1]
        self.relevance_vector = self.training_data
        self.phi = self.phi_function(self.training_data, self.training_data)

        self.alphas = np.array([1 / (self.training_data.shape[0] + 1)] * (self.training_data.shape[0] + 1))
        self.weight = np.array([1 / (self.training_data.shape[0] + 1)] * (self.training_data.shape[0] + 1))

        max_training_iterations = 1000
        threshold = 1e-3
        for i in tqdm(range(max_training_iterations)):
            self.alphas_old = np.copy(self.alphas)

            self.update_weights()
            y = self.y_function(self.weight, self.phi)
            beta = self.beta_matrix_function(y)
            sigma = self.sigma_function(self.phi, beta, self.alphas)

            gammas = self.gamma_function(self.alphas, sigma)
            self.alphas = self.recalculate_alphas_function(gammas, self.weight)

            # self.prune()
            self.prune()

            difference = np.amax(np.abs(self.alphas - self.alphas_old))  # Need to change this
            if difference < threshold:
                print("Training done, it converged. Nr iterations: " + str(i + 1))
                break

    def predict(self, data=[], use_predefined_training=False):
        if data == []:
            if use_predefined_training:
                data = self.training_data
            else:
                data = self.test_data

        phi = self.phi_function(data, self.relevance_vector)
        y = self.y_function(self.weight, phi)
        pred = np.where(y > 0.5, 1, 0)
        self.prediction = pred
        return pred

    def plot(self, data=[], target=[]):
        if data == [] and target == []:
            data = self.test_data
            target = self.test_labels
        else:
            target[target == -1] = 0  # Sanitize labels, some use -1 and some use 0

        # Format data so we can plot it
        print("Will start plotting")
        target_values = np.unique(target)
        data_index_target_list = [0] * len(target_values)
        for i, c in enumerate(target_values):  # Saving the data index with its corresponding target
            data_index_target_list[i] = (c, np.argwhere(target == c))

        h = 0.01  # step size in the mesh
        # create a mesh to plot in
        x_min, x_max = data[:, 0].min(), data[:, 0].max()  # Gets the max and min value of x in the data
        y_min, y_max = data[:, 1].min(), data[:, 1].max()  # Gets the max and min value of y in the data
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))  # Creates a mesh from max and min with step size h

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        print("Calculating the prediction and will plot, this might take a while...")
        data_mesh = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(data_mesh)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)

        colors = ["rx", "bx"]
        plt.figure(figsize=(12, 6))
        plt.title('')
        plt.contour(xx, yy, Z, cmap=plt.get_cmap('Greys'))
        for i, c in enumerate(data_index_target_list):
            plt.plot(data[c[1], 0], data[c[1], 1], colors[i], label="Target: " + str(c[0]))
        plt.scatter(self.relevance_vector[:, 0], self.relevance_vector[:, 1], c='black', marker='o', s=80, alpha=0.5)
        plt.xlabel("")
        plt.ylabel("")
        plt.legend()
        plt.grid('on')
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.show()

    def get_prediction_error_rate(self, predicted_targets=[], true_targets=[], use_predefined_training=False):
        if predicted_targets == [] and true_targets == []:
            predicted_targets = self.prediction
            if use_predefined_training:
                true_targets = self.training_labels
            else:
                true_targets = self.test_labels

        nr_correct = 0
        for i in range(len(predicted_targets)):
            if predicted_targets[i] == true_targets[i]:
                nr_correct += 1
        return 1 - nr_correct / len(predicted_targets)
