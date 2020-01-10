import numpy as np

import rvm_classification

test_data = np.loadtxt("datasets/ripley/ripley_test_data_other_guys.asc")
test_target = np.loadtxt("datasets/ripley/ripley_test_labels_other_guys.asc")

train_data = np.loadtxt("datasets/ripley/ripley_train_data_other_guys.asc")
train_target = np.loadtxt("datasets/ripley/ripley_train_labels_other_guys.asc")

# test_data = np.loadtxt("datasets/titanic/titanic_test_data_1.asc")
# test_target = np.loadtxt("datasets/titanic/titanic_test_labels_1.asc")
#
# train_data = np.loadtxt("datasets/banana/banana_train_data_1.asc")
# train_target = np.loadtxt("datasets/banana/banana_train_labels_1.asc")


rvc = rvm_classification.RVC()

# rvc.set_predefined_training_data("banana")
rvc.set_training_data(train_data, train_target)
rvc.fit()
rvc.plot(test_data, test_target)
# rvc.plot()


