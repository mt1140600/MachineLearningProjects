import numpy as np
import csv
import math
from sklearn import mixture
v = '/home/gauss.csv'
# c = mixture.GMM(n_components=6, covariance_type='full')
# c.fit(data)
# print('kk')
# clusterIdentifiers = c.predict(data)
# print('dddddddddd')
# print(c.covars_)
# print('dddddddddd')
# with open('gauss1.csv', 'w', encoding='utf8', newline='') as svfile:
#     spmwriter = csv.writer(svfile)
#     for i in range(0, len(clusterIdentifiers)):
#         spmwriter.writerow([clusterIdentifiers[i]])
# print(clusterIdentifiers)



clusters = 6
x = [3,3,3,3,2,3,3,3,3,3,3,3,3,3,3,3,4,4,3,3,3,0,3,3,3,4,3,3,3,3,3,2,3,3,3,3,0,3,4,3,3,3,3,3,3,0,0,3,3,3,3,3,0,3,3,4,3,3,3,0,3,0,4,0,3,2,3,3,3,2,3,3,2,3,3,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,3,2,3,3,3,3,3,0,3,3,3,3,3,3,3,3,4,0,3,3,3,3,4,3,0,3,0,3,3,4,0,3,3,3,0,3,3,4,3,3,3,0,3,3,4,3,3,3,3,3,3,2,3,0,3,4,2,3,3,3,3,3,0,0,2,3,3,3,3,0,3,3,3,3,3,3,4,3,3,3,3,3,3,3,3,3,3,3,3,3,0,3,3,3,3,4,3,4,0,3,3,3,0,3,3,2,3,4,2,3,3,3,3,2,0,3,3,3,3,0,2,3,3,3,0,3,2,3,2,3,2,0,3,0,3,3,4,3,3,3,0,3,3,3,2,3,3,0,3,3,3,3,3,2,3,3,3,4,3,3,4,2,3,4,3,2,3,0,4,3,3,3,0,3,3,3,3,3,3,3,3,3,4,3,3,3,3,3,3,0,3,3,4,3,3,2,3,3,4,0,4,3,3,0,3,0,3,3,0,4,0,3,0,3,3,3,3,3,3,3,3,3,3,3,3,2,4,3,0,3,0,4,2,3,3,3,3,0,3,3,3,0,2,3,3,3,0,3,3,3,0,3,3,4,3,2,3,3,3,3,3,3,3,0,2,3,2,3,3,2,3,4,4,3,0,3,3,3,3,2,3,3,0,0,0,3,3,2,3,3,3,3,0,2,3,3,3,0,3,3,2,3,3,0,3,3,4,3,3,4,3,3,3,3,3,4,2,3,3,3,3,3,3,3,3,3,3,0,4,3,3,3,3,0,0,3,3,0,3,3,3,3,3,2,3,3,3,0,3,3,3,3,3,3,3,3,3,0,3,2,4,3,3,3,2,3,3,3,3,3,3,3,3,0,3,0,3,4,2,3,3,3,4,2,3,3,2,3,3,3,0,2,3,4,3,3,3,3,3,2,4,2,3,3,0,3,3,3,3,3,4,0,0,2,3,3,3,3,3,0,2,2,2,3,3,0,4,3,0,3,3,3,0,3,0,3,3,3,3,2,3,0,3,3,3,3,3,3,3,3,3,2,3,3,2,3,0,3,3,3,3,3,3,3,4,3,3,3,4,0,3,3,3,3,3,3,3,3,2,3,0,3,3,3,3,2,2,0,4,0,0,3,3,3,4,0,0,3,3,3,3,3,0,2,4,3,3,3,3,3,3,3,2,3,3,3,3,3,2,3,3,4,3,0,3,3,3,3,4,3,3,3,0,3,2,3,3,0,4,3,3,3,3,0,4,2,3,3,3,3,3,3,3,3,3,3,0,3,2,3,3,3,3,3,3,0,3,3,4,3,3,2,3,3,0,3,3,3,3,3,3,3,3,3,3,3,0,3,3,3,3,3,4,3,3,4,3,3,3,3,3,3,0,3,0,0,3,3,3,4,2,4,3,3,2,3,3,3,4,2,0,3,0,3,3,4,3,0,3,3,3,3,3,3,3,3,0,4,3,3,3,0,2,3,3,3,3,3,3,3,3,2,3,2,3,0,3,3,3,4,3,3,3,3,2,3,4,3,3,3,3,3,0,3,4,3,3,3,2,0,3,2,3,2,3,3,0,0,3,3,3,3,3,3]

def save_model(file_name, model_list):
    import pickle
    with open(file_name, 'wb') as fid:
        pickle.dump(model_list, fid)


# Loading the model from the pickle file
def load_model(file_name):
    import pickle
    with open(file_name, 'rb') as fid:
        model = pickle.load(fid)
    return model


def strToInt(row):
    arr=[]
    arr.append(float(row[0]))
    arr.append(float(row[1]))
    arr = np.array(arr)
    return (arr)


def calcProb(data, means, covariance):
    data = np.array(data)
    means = np.array(means)
    covariance = np.array(covariance)
    # if(np.linalg.det(covariance)==0):
    #     covariance = np.array([[0.6,0.5],[0.5,0.7]])
    p = data-means
    p1 = np.matrix(p)
    y1 = np.dot(np.linalg.inv(covariance),p1.transpose())
    y = np.dot(p,y1)
    norm = 2*(math.pi)*(math.sqrt(abs(np.linalg.det(covariance))))
    prob = (1/norm)*math.exp((-0.5)*(y))
    return prob


def maxProb(data, means, covariances, mixingCoeffs):
    probs = []
    maxP = 0
    for i in range(0,clusters):
        x =  (mixingCoeffs[i])*calcProb(data, means[i], covariances[i])/790
        probs.append(x)
        if(x>maxP):
            maxP = x
    return [maxP,probs.index(maxP)]


def calcMix(n,arr):
    count=0
    for i in range(len(arr)):
        if(arr[i]==n):
            count=count+1
    return count


def calcMean(data):
    # print(len(data))
    if(len(data)==0):
        return np.array([0.,0.])
    x = np.array([0.,0.])
    for i in range(0,len(data)):
        x = x + data[i]
    x = x/len(data)
    return x



def calcMeanCluster(dataset, clusterIdentifier):

    clusterMean = []
    for i in range(0,clusters):
        clusterSet=[]
        for j in range(0, len(clusterIdentifier)):
            if(clusterIdentifier[j]==i):
                clusterSet.append(dataset[j])
        clusterMean.append(calcMean(clusterSet))

    return clusterMean



def GaussMixtureMeans(datset, means ,covarianes, mixingCoeffs):

    clusterIdentifier = []
    for i in range(0,len(datset)):
        j = maxProb(datset[i], means, covarianes, mixingCoeffs)[1]
        clusterIdentifier.append(j)
    return clusterIdentifier


if __name__=='__main__':

    csv_trainlabel = "train_data.csv"
    data = []
    with open(csv_trainlabel) as ifile:
        read = csv.reader(ifile)
        for row in read:
            # print(row)
            # print(strToInt(row))
            data.append(strToInt(row))
    data = np.array(data)
    clusterMeans = np.array([[-1.82517034, -0.40389968],[1.10371788,  0.58891785],[0.86267684, -1.1656794],[-1.93101753 ,0.77641046],[1.06681984,  0.46228307],[-0.66701728, 0.66295488]])

    v = []
    for j in range(0, clusters):
        n = calcMix(j, x)
        if (n == 0):
            v.append(np.array([[1., 0.], [0., 1.]]))
        else:
            o = np.array([[0., 0.], [0., 0.]])
            number = 0
            for k in range(0, len(x)):
                if (x[k] == j):
                    number = number + 1
                    p = (data[k] - clusterMeans[j])
                    p = np.matrix(p)
                    o = o + np.dot(p.transpose(), p)
            o = o / number
            v.append(o)

    clusterCovariances =v
    mixingCoefficients = []

    for i in range(0,clusters):
        mixingCoefficients.append(1/clusters)

    clusterIdentifiers = GaussMixtureMeans(data, clusterMeans, clusterCovariances, mixingCoefficients)
    print(clusterIdentifiers)

    i=0
    m = [clusterMeans, clusterCovariances, mixingCoefficients]
    save_model('model.pkl', m)
    while(i<30):

        print(i)
        clusterMeans = calcMeanCluster(data, clusterIdentifiers)
        for j in range(0,clusters):
            n  = calcMix(j,clusterIdentifiers)
            if(n==0):
                clusterCovariances[j]=(np.array([[1., 0.], [0., 1.]]))
            else:
                x = np.array([[0., 0.], [0., 0.]])
                number = 0
                for k in range(0,len(clusterIdentifiers)):
                    if(clusterIdentifiers[k]==j):
                        number = number+1
                        p = (data[k]-clusterMeans[j])
                        p = np.matrix(p)
                        x = x + np.dot(p.transpose(),p)
                        # print(np.dot((data[k]-clusterMeans[j]),(data[k]-clusterMeans[j]).transpose()))
                x = x/number
                # print(x)
                clusterCovariances[j]=x

        for j in range(0,clusters):
            mixingCoefficients[j] = calcMix(j,clusterIdentifiers)
        print(mixingCoefficients)
        # print('coeffs')
        clusterIdentifiers = GaussMixtureMeans(data, clusterMeans, clusterCovariances, mixingCoefficients)
        i = i+1

        print(clusterIdentifiers)
    with open('gauss.csv', 'w', encoding='utf8', newline='') as svfile:
        spmwriter = csv.writer(svfile)
        for i in range(0, len(clusterIdentifiers)):
            spmwriter.writerow([clusterIdentifiers[i]])

