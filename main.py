import rvm_classification
import numpy as np

test_data = np.loadtxt("datasets/banana_test_data_dev.asc")
test_taraget = np.loadtxt("datasets/banana_test_labels_dev.asc")

train_data = np.loadtxt("datasets/banana_train_data_dev.asc")
train_taraget = np.loadtxt("datasets/banana_train_labels_dev.asc")

classifier = rvm_classification.RVM_Classifier()
classifier.train(train_data, train_taraget)
# classifier.predict(test_data)
classifier.plot(test_data, test_taraget)
