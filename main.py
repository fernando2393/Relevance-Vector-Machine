import rvm_classification

# How to load non predefined data
# test_data = np.loadtxt("datasets/ripley_test_data.asc")
# test_target = np.loadtxt("datasets/ripley_test_labels.asc")
#
# train_data = np.loadtxt("datasets/ripley_train_data.asc")
# train_target = np.loadtxt("datasets/ripley_train_labels.asc")

classifier = rvm_classification.RVM_Classifier()
# classifier.set_training_data(train_data, train_target)

# Takes the dataset and optional index and set training, test values
classifier.set_predefined_training_data("banana")
classifier.train()
# classifier.predict(test_data)
classifier.plot()
