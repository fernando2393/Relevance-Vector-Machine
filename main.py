import rvm_classification
import numpy as np

# Small data set used for development
# test_data = np.loadtxt("datasets/banana_test_data_dev.asc")
# test_taraget = np.loadtxt("datasets/banana_test_labels_dev.asc")
#
# train_data = np.loadtxt("datasets/banana_train_data_dev.asc")
# train_taraget = np.loadtxt("datasets/banana_train_labels_dev.asc")

# Banana 1, whole data set
# test_data = np.loadtxt("datasets/banana_test_data_1.asc")
# test_taraget = np.loadtxt("datasets/banana_test_labels_1.asc")
#
# train_data = np.loadtxt("datasets/banana_train_data_1.asc")
# train_taraget = np.loadtxt("datasets/banana_train_labels_1.asc")

# Same as the people
test_data = np.loadtxt("datasets/ripley_test_data.asc")
test_target = np.loadtxt("datasets/ripley_test_labels.asc")

train_data = np.loadtxt("datasets/ripley_train_data.asc")
train_target = np.loadtxt("datasets/ripley_train_labels.asc")

classifier = rvm_classification.RVM_Classifier()
classifier.train(train_data, train_target)
# classifier.predict(test_data)
classifier.plot(test_data, test_target)
