# useful import statements
from __future__ import print_function
from sklearn.svm import libsvm
import math
from sklearn.datasets import load_svmlight_file
import numpy as np
import csv

# some useful library functions to load test data file
np.set_printoptions(threshold=np.inf)
f = open("test_set.csv")
data = np.loadtxt(f)

#Dumping model in a pickle file
def save_model(file_name, model_list):
    import pickle
    with open(file_name, 'wb') as fid:
        pickle.dump(model_list, fid)
 
#Loading the model from the pickle file
def load_model(file_name):
    import pickle
    with open(file_name, 'rb') as fid:
        model = pickle.load(fid)
    return model



# Calculation of mean of an array
def calcMean(arr):
    x = 0
    for i in range(0, len(arr)):
        x = x + arr[i]
    return x/len(arr)


# Here the array is normalised using array = (array - mean)/variance
def normaliseData(arr):
    mean = calcMean(arr)
    variance = calcMean(np.square(arr)) - mean*mean
    return (arr-mean)/math.sqrt(variance)

# Training set and labels are loaded here
X,Y = load_svmlight_file(f='train_raw', n_features=16, multilabel=False, zero_based='auto', query_id=False, dtype=np.float64)
X = X.toarray()

# The original training and test labels are normalised here
for j in range(0,len(X)):
    X[j] = normaliseData(X[j])
for j in range(0,len(data)):
    data[j] = normaliseData(data[j])


# SVM model is trained here using libsvm in-built library functions. Kernel typr is polynomial here and its degree is kept 4.
# All parameters are kept for making the most optimal fit for the data
[support, sv, nsv, coeff, intercept, proba, probb, fit_status] = libsvm.fit(X, Y, svm_type=0, kernel='poly', degree=4,
        gamma=0.093, coef0=0, tol=0.001, C=1, nu=0.5, max_iter=-1, random_seed=0)

m = [support, sv, nsv, coeff, intercept, proba, probb]
save_model('model.pkl',m)

[support_, sv_, nsv_, coeff_, intercept_, proba_, probb_] = load_model('model.pkl')
# Predictions are made on the test dataset using the hyper parameters trained on training dataset
dec_values = libsvm.predict(data, support_, sv_, nsv_, coeff_, intercept_, proba_, probb_, svm_type=0,kernel='poly', degree =4,
                            gamma=0.093, coef0=0)
dec_values.astype(int)

# Predictions are written to a csv file named result.csv
j=0
with open('result.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    for j in range(0,len(dec_values)):
        spamwriter.writerow([int(dec_values[j])])

        Host ID: 90489ad33ff3 74e6e22945bf         Release: R2015b         Login Name: root
        [ 58 139  66  61  76  90  43  96 164 143]