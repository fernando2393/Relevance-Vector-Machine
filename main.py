import rvm_classification

relevance_vectors = []
errors = []
nr_iterations = 1
data_set = "titanic"
for i in range(nr_iterations):
    print("data_set {n}".format(n=i))
    rvc = rvm_classification.RVM_Classifier()
    rvc.set_predefined_training_data(data_set, data_set_index=i + 1)
    rvc.fit()
    prediction = rvc.predict(use_predefined_training=False)
    errors.append(rvc.get_prediction_error_rate(use_predefined_training=False))
    relevance_vectors.append(rvc.get_nr_relevance_vectors())
    # rvc.plot()

print("Result for data set: " + data_set + " for the " + str(nr_iterations) + " indexes")
print("Errors for each data set index")
print(*errors, sep=", ")

print("Number of relevance vectors for each data set index")
print(*relevance_vectors, sep=", ")

print("Average error for data set: " + data_set + " is " + str(sum(errors) / nr_iterations))
print("Average number relevance vectors for data set: " + data_set + " is " + str(sum(relevance_vectors) / nr_iterations))
