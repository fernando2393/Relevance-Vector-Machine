import numpy as np

import rvm_classification

test_data = np.loadtxt("datasets/ripley/ripley_test_data_other_guys.asc")
test_target = np.loadtxt("datasets/ripley/ripley_test_labels_other_guys.asc")

train_data = np.loadtxt("datasets/ripley/ripley_train_data_other_guys.asc")
train_target = np.loadtxt("datasets/ripley/ripley_train_labels_other_guys.asc")

rvc = rvm_classification.RVM_Classifier()

# test_data, test_target = rvc.get_nr_random_samples(train_data, train_target, 250)

# rvc.set_predefined_training_data("image")
rvc.set_training_data(train_data, train_target)
# rvc.set_training_data(test_data, test_target)
rvc.fit()
# rvc.plot(test_data, test_target)
rvc.plot(train_data, train_target)
# rvc.plot()
