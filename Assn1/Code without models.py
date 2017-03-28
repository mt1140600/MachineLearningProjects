# importing of useful libraries
import numpy as np
import csv
import math

# initialisation of variables
# features dimension is 16 and then a bias is considered
# learning rat for gradient descent
featureDimension = 17
hiddenLayerSize = 200
classes = 26
learningRate = 0.004
regRate = 0.000001
testAccuracy = 0
trainingAccuracy = 0
maxtestAccuracy = 0
maxtrainingAccuracy = 0
np.set_printoptions(threshold=np.inf)
# calcultion of cross entropy const function
# measuredValues are the ones which are calculated by the model during its training for different examples
# Basically measured values are the matrix of softmax probabilities for different output classes
# actualvalue is the value that is the actual output for a particular example
def costFunction(measuredValue, actualValue):
    return -(math.log10(measuredValue.item(actualValue-1)))


# This is the result of backpropagation of error when done upto the hidden layer to coreect the second layer
# weights. The binary matrix declared here denotes the indicator function for the probablities measure using softmax
#  function. It gives one for the class which is actually the expected output and zero for rest of the classes
# neuronValues are the values obtained in the hidden layer. actualvalue and measuredvalue notations are same as before
def minErrorAtHidden(measuredValue, actualValue, neuronValues):
    # print(actualValue)
    binaryMatrix = np.zeros(26)
    binaryMatrix[actualValue-1] = 1
    return np.dot(neuronValues.transpose(), (measuredValue.transpose()-binaryMatrix))

# This is the result of back-propagating of error when done upto first layer to correct first layer weights
# weights2 are the weights of second layer connecting hidden layer to output classes. binaryMatrix1 is again the indicator
# function same as before
def minErrorAtFirst(features, weights2, actualValue, measuredValue):
    binaryMatrix1 = np.zeros(26)
    binaryMatrix1[actualValue-1] = 1
    x = np.dot(measuredValue.transpose()-binaryMatrix1, weights2.transpose())
    return np.dot(features, x)


# this function is the implementation of softmax function on a matrix
def normaliseSoftmax(arr):
    arr = np.exp(arr)
    arr = arr/(np.sum(arr, axis=None, keepdims=True))
    return arr

# this function is the implementation of sigmoid function on a matrix
def normaliseSigmoid(arr):
    arr = np.exp(arr)
    arr = arr/(1+np.exp(arr))
    return arr




# This is the most important function which trains our network.
def trainNetwork(features, weights1, weights2, output):

    #Calculating hidden layer values and then normalising it
    neuronValues = np.dot(features.transpose(), weights1)

    #ReLu activation used
    neuronValues = np.maximum(0, neuronValues)
    # bias added
    np.append([[1]], neuronValues, axis=None)
    # Calculating final values for all classes and then normalising it
    finalOutput = np.dot(weights2.transpose(), neuronValues.transpose())

    # softmax implemented
    finalOutput = normaliseSoftmax(finalOutput)

    # storing of weights in temporary variables
    temp2 = weights2 - learningRate * minErrorAtHidden(finalOutput, output, neuronValues) - regRate*weights2
    temp1 = weights1 - learningRate * minErrorAtFirst(features, weights2, output, finalOutput) - regRate*weights1
    return [temp1, temp2]

if __name__ == '__main__':

    # initialisation of weights in a random matrix
    np.random.seed(seed=2)
    weights1 = np.random.randn(featureDimension, hiddenLayerSize)
    weights2 = np.random.randn(hiddenLayerSize, classes)

    # storing of weights in temporary variables. These are used since weights are updated only and only if the
    # accuracy on cross validation set is increasing and also training set is increasing. this is to avoid overfitting
    # of the data
    temp = weights1
    temp1 = weights2

    # storing the addresses of the files to be read
    csv_label = "/home/raghav/Desktop/train_labels.csv"
    csv_path = "/home/raghav/Desktop/train_data.csv"
    csv_testlabel = "/home/raghav/Desktop/test_data.csv"

    # initialisation of variables
    actualValues = [] # expected output stored in this from training labels
    i = 0
    error = 0  # denotes the cost on training set
    cross = 0  # cost on validation test set
    Olderror = 500000000000 # training cost of previous epoch,initialsised with random value but updated later
    OldCross = 5000000000   # validation test cost of previous epoch,initialsised with random value but updated later

    # actual values are read and stored in this array
    with open(csv_label) as ifile:
        read = csv.reader(ifile)
        for row in read:
            actualValues.append(float(row[0]))
    j = 0

    # main function starts here, error cost are stored in a csv file
    with open('errorCost.csv', 'w',encoding='utf8',newline='') as svfile:
        spmwriter = csv.writer(svfile)
        # epoch started
        for l in range(1, 100):

            # read feartures from file and store them in a matrix along with a bias
            with open(csv_path) as ifile:
                read = csv.reader(ifile)
                for row in read:

                    features = np.matrix(
                        [[1], [row[0]], [row[1]], [row[2]], [row[3]], [row[4]], [row[5]], [row[6]], [row[7]], [row[8]],
                         [row[9]]
                            , [row[10]], [row[11]], [row[12]], [row[13]], [row[14]], [row[15]]], dtype=float)
                    #model is trained training set
                    if(j<13200):
                        z = trainNetwork(features, weights1, weights2, actualValues[j])

                        # storage of updated weights
                        weights1 = z[0]
                        weights2 = z[1]
                        # outputs are calculated from the new updated weights
                        neurons = np.dot(features.transpose(), weights1).transpose()
                        neurons = np.maximum(0, neurons)

                       # bias added
                        np.append([[1]],neurons,axis=None)

                        finalOutput = np.dot(weights2.transpose(), neurons)
                        finalOutput = normaliseSoftmax(finalOutput)
                        h = max(finalOutput)
                        # final class is calculated for which the probabilty is highest
                        for p in range(0,26):
                            if(finalOutput.item(p)==h):
                                break
                        # if the calculated and expected clases match the accuracy parameter is increased
                        if (p + 1 == actualValues[j]):
                            trainingAccuracy += 1
                        # finally cost is calculated, regularisation is added in last
                        error += costFunction(finalOutput, actualValues[j])

                    else:
                        neurons = np.dot(features.transpose(), weights1).transpose()
                        # ReLu activation function
                        neurons = np.maximum(0, neurons)
                        np.append([[1]], neurons, axis=None)
                        # finaloutput is the matrix of output for different classes
                        # activation function used here is softmax function

                        finalOutput = np.dot(weights2.transpose(), neurons)
                        # softmax activation function implemented
                        finalOutput = normaliseSoftmax(finalOutput)

                        h1 = max(finalOutput)
                        for p in range(0, 26):
                            if (finalOutput.item(p) == h1):
                                break

                        if (p + 1 == actualValues[j]):
                            testAccuracy += 1

                        cross += costFunction(finalOutput, actualValues[j])
                    j = j+1
                # regularisation added
            error = error + 0.5*regRate*np.sum(np.dot(weights1, weights1.transpose()), axis=None)+0.5*regRate*np.sum(np.dot(weights2, weights2.transpose()), axis=None)

            # debug statements

            # print(l)
            print(testAccuracy/18)
            # print(maxtestAccuracy)
            # print(trainingAccuracy/132)
            # print(maxtrainingAccuracy)
            # print()
            spmwriter.writerow([error])

            j = 0
            # condition for next epoch to run is checked
            if (testAccuracy<maxtestAccuracy):
                break
            # parameters updated for next epoch
            Olderror = error
            if(maxtestAccuracy<testAccuracy):
                maxtestAccuracy = testAccuracy
            error = 0
            cross = 0
            testAccuracy = 0
            trainingAccuracy = 0
    # weights are stored in an csv file
    with open('weights.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow([weights1])
        spamwriter.writerow([weights2])
    z= 0
    # test set calculations are started here. first the features are readed and stored in matrix
    # and rest of procedure is same as for training the neural network
    with open(csv_testlabel) as ifile:
        read = csv.reader(ifile)
        with open('result.csv', 'w') as csvfile:
            spamwriter = csv.writer(csvfile)
            for row in read:

                features1 = np.matrix(
                    [[1], [row[0]], [row[1]], [row[2]], [row[3]], [row[4]], [row[5]], [row[6]], [row[7]],
                     [row[8]], [row[9]]
                        , [row[10]], [row[11]], [row[12]], [row[13]], [row[14]], [row[15]]], dtype=float)

                # calculation of results
                neurons1 = np.dot(features1.transpose(), weights1).transpose()
                neurons1 = np.maximum(0, neurons1)
                # bias added
                np.append([[1]], neurons1, axis=None)
                finalOutput1 = np.dot(weights2.transpose(), neurons1)
                finalOutput1 = normaliseSoftmax(finalOutput1)
                h1 = max(finalOutput1)

                for p in range(0, 26):
                    if (finalOutput1.item(p) == h1):
                        break
                c = int(p+1)
                spamwriter.writerow([c])
