import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


class SVM_Classifier(object):
    def __init__(self, gamma):
        
        self.gamma = gamma
        self.relevance_vector = None
        self.prediction = None      
        self.predict_error = []
        self.total_support_vector = []

    def get_prediction_error_rate(self,predicted_targets=[], true_targets=[]):
        nr_correct = 0
        for i in range(len(predicted_targets)):
            if predicted_targets[i] == true_targets[i]:
                nr_correct += 1
        return 1 - nr_correct / len(predicted_targets)
    

    def classification(self,train_data, train_labels, test_data, test_labels):
        self.test_data = test_data
        self.test_labels = test_labels
        self.train_data = train_data
        self.train_labels = train_labels
         
        np.random.seed(0)
        c_range = np.array([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        #c_range = np.array([0.001])
        k_fold = 5

        score = np.zeros(len(c_range))
        for c in range(len(c_range)):
            svclassifier = SVC(kernel="rbf", gamma=self.gamma, C=c_range[c])
            score[c] = np.mean(cross_val_score(svclassifier, train_data, train_labels, cv=k_fold))
            

        c = c_range[np.argmax(score)]
        #print("this is  the best c {c}".format(c = c))
        svclassifier = SVC(kernel="rbf", gamma=self.gamma, C=c)

        svclassifier.fit(train_data, train_labels)
        y_pred = svclassifier.predict(test_data)
        predicted_error = self.get_prediction_error_rate(y_pred, test_labels)    
            

        self.relevance_vector = svclassifier.support_vectors_

        #### NUMBER OF SUPPORT VECTORS
        self.total_support_vector.append(sum(svclassifier.n_support_))
        self.predict_error.append(self.get_prediction_error_rate(y_pred, test_labels))
        
        return self.total_support_vector, self.predict_error, svclassifier



    def plot(self, classifier, data=[], target=[]):
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
        Z = classifier.predict(data_mesh)
        

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
