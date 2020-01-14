import rvm_classification
import svm_classification
import numpy as np

relevance_vectors = []
support_vectors = []
rvm_error = []
svm_error = []

r = None # None
gamma = "auto" # "auto"

nr_iterations = 10
data_set = "titanic"
for i in range(nr_iterations):
    print("\n data_set {n}".format(n=(i+1)))
    rvc = rvm_classification.RVM_Classifier(r)
    rvc.set_predefined_training_data(data_set, data_set_index=i + 1)
    rvc.fit()
    prediction = rvc.predict(use_predefined_training=False)
    rvm_error.append(rvc.get_prediction_error_rate(use_predefined_training=False))
    relevance_vectors.append(rvc.get_nr_relevance_vectors())
    
    training_data, training_labels, test_data, test_labels = rvc.saving_dataset()
    svc = svm_classification.SVM_Classifier(gamma)
    sv, error,svclassifier = svc.classification(training_data, training_labels, test_data, test_labels)
    support_vectors.append(sv)
    svm_error.append(error)
    
    if data_set == "ripley":
        rvc.plot()
        svc.plot(classifier=svclassifier)
        

print("\n Relevance Vector Machine\n ")

print("Result for data set: " + data_set + " for the " + str(nr_iterations) + " indexes")
print("error for each data set index")
print(*rvm_error, sep=", ")

print("Number of relevance vectors for each data set index")
print(*relevance_vectors, sep=", ")

print("Average error for data set: " + data_set + " is " + str(sum(rvm_error) / nr_iterations))
print("Average number relevance vectors for data set: " + data_set + " is " + str(sum(relevance_vectors) / nr_iterations))


print("\n Support Vector Machine\n ")

print("Result for data set: " + data_set + " for the " + str(nr_iterations) + " indexes")
print("error for each data set index")
print(*svm_error, sep=", ")

print("Number of support vectors for each data set index")
print(*support_vectors, sep=", ")

print("Average error for data set: " + data_set + " is " + str(sum(np.array(svm_error)) / nr_iterations))
print("Average number support vectors for data set: " + data_set + " is " + str(sum(np.array(support_vectors)) / nr_iterations))

