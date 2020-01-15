import rvm_classification
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

relevance_vectors = []
errors = []
nr_iterations = 2
data_set = "images"
usps_predictions = []
rvc = rvm_classification.RVM_Classifier()
rvc.set_usps_data()
for i in range(nr_iterations):
    rvc.set_predefined_training_data(data_set, data_set_index=i + 1)
    rvc.fit()
    usps_predictions.append(rvc.predict(use_predefined_training=False))

usps_predictions = np.array(usps_predictions)
clean_predictions = []
for row in range(usps_predictions.shape[1]):
    clean_predictions.append(stats.mode(usps_predictions[:,row]).mode[0])

rvc.set_prediction_usps(clean_predictions)
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
