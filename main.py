import rvm_classification

# test_data = np.loadtxt("datasets/ripley/ripley_test_data_other_guys.asc")
# test_target = np.loadtxt("datasets/ripley/ripley_test_labels_other_guys.asc")
#
# train_data = np.loadtxt("datasets/ripley/ripley_train_data_other_guys.asc")
# train_target = np.loadtxt("datasets/ripley/ripley_train_labels_other_guys.asc")

rvc = rvm_classification.RVM_Classifier()

# test_data, test_target = rvc.get_nr_random_samples(train_data, train_target, 250)

# rvc.set_predefined_training_data("breast-cancer", data_set_index=1)
# # rvc.set_training_data(train_data, train_target)
# # rvc.set_training_data(test_data, test_target)
# rvc.fit()
# # rvc.plot(test_data, test_target)
# # rvc.plot(train_data, train_target)
# # rvc.plot()
# prediction = rvc.predict(use_predifined_training=False)
# error_rate = rvc.get_prediction_error_rate(use_predefined_training=False)
# print("Error rate is: " + str(error_rate))
#
# ok =rvc.get_nr_relevance_vectors()



relevance_vectors = []
errors = []
nr_iterations = 10
data_set = "banana"
for i in range(nr_iterations):
    rvc = rvm_classification.RVM_Classifier()
    # print("Running training on data set: " + data_set + " index: " + str(i+1))
    rvc.set_predefined_training_data(data_set, data_set_index=i+1)
    rvc.fit()
    prediction = rvc.predict(use_predifined_training=False)
    errors.append(rvc.get_prediction_error_rate(use_predefined_training=False))
    relevance_vectors.append(rvc.get_nr_relevance_vectors())
    #rvc.plot()


print("Result for data set: " + data_set + " for the " + str(nr_iterations) + " indexes")
print("Errors for each data set index")
print(*errors, sep = ", ")

print("Number of relevance vectors for each data set index")
print(*relevance_vectors, sep = ", ")

print("Average error for data set: " + data_set + " is " + str(sum(errors)/nr_iterations))
print("Average number relevance vectors for data set: " + data_set + " is " + str(sum(relevance_vectors)/nr_iterations))
