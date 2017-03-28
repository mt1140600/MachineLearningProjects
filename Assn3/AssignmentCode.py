import numpy as np
import csv
import math

clusters = 6
# data[l] = np.array([[float(row[0]),float(row[1])]])
def strToInt(row):
    arr=[]
    arr.append(float(row[0]))
    arr.append(float(row[1]))
    return (arr)

def calcDistance(p1, p2):
    # print(p2)
    # print(p1)
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

def calcMinDistance(data, clusterCentres):
    minDis = 99999999999999
    distances = []
    # print('ff')
    # print(clusterCentres[0])
    for i in range(0,len(clusterCentres)):
        # print(clusterCentres[i])
        x = calcDistance(data, clusterCentres[i])
        distances.append(x)
        if(x<minDis):
            minDis = x
    return [minDis, distances.index(minDis)]

def calcMean(data):
    # print(len(data))
    if(len(data)==0):
        return np.array([0.,0.])
    x = np.array([0.,0.])
    for i in range(0,len(data)):
        x = x + data[i]
    x = x/len(data)
    return x

def normaliseData(arr):
    mean = calcMean(arr)
    mean = np.array(mean)
    return (arr - mean)

def kMeansCluster(dataset, clusterCentres):

    clusterIdentifier = []
    for i in range(0, len(dataset)):
        j = calcMinDistance(dataset[i], clusterCentres)[1]
        clusterIdentifier.append(j)

    return clusterIdentifier


def calcMeanCluster(dataset, clusterIdentifier, clusters):

    clusterMean = []
    for i in range(0,clusters):
        clusterSet=[]
        for j in range(0, len(clusterIdentifier)):
            if(clusterIdentifier[j]==i):
                clusterSet.append(dataset[j])
        clusterMean.append(calcMean(clusterSet))
    return clusterMean


if __name__=='__main__':

    # np.random.seed(seed=2)
    csv_trainlabel = "train_data.csv"
    data = []
    with open(csv_trainlabel) as ifile:
        read = csv.reader(ifile)
        for row in read:
            # print(row)
            # print(strToInt(row))
            data.append(strToInt(row))
    data = np.array(data)
    print(data)
    # data = normaliseData(data)

    clusterCentres = [[-1.82517034, -0.40389968], [1.10371788,  0.58891785], [0.86267684 ,-1.1656794], [-1.93101753, 0.77641046],[1.06681984 , 0.46228307], [-0.66701728, 0.66295488]];
    # print(clusterCentres)
    clusterIdentifiers = kMeansCluster(data,clusterCentres)
    # print(clusterCentres)
    print(clusterIdentifiers)


    i=0
    while(i<120):
        print(i)
        clusterCentres = calcMeanCluster(data, clusterIdentifiers, clusters)
        clusterIdentifiers = kMeansCluster(data,clusterCentres)
        print(clusterIdentifiers)
        i=i+1
    print("kjbdb")
    with open('Kmeans.csv', 'w', encoding='utf8', newline='') as svfile:
        spmwriter = csv.writer(svfile)
        for i in range(0, len(clusterIdentifiers)):
            spmwriter.writerow([clusterIdentifiers[i]])
    print(clusterIdentifiers)


























































































