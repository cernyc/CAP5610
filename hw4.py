import pandas
import numpy as np
from numpy.linalg import norm
import sklearn
from sklearn.cluster import KMeans
from math import sqrt, floor

dp = [[3, 5], [3, 4], [2, 8], [2, 3], [6, 2], [6, 4], [7, 3], [7, 4], [8, 5], [7, 6]]

#centr = [[4, 6], [5, 4]]
#centr = [[3, 3], [8, 3]]
centr = [[3, 2], [4, 8]]

def manDist (point1, point2):
    if point1[0] < point2[0]:
        x = (point2[0] - point1[0])
    else:
        x = (point1[0] - point2[0])
    if point1[1] < point2[1]:
        y = (point2[1] - point1[1])
    else:
        y = (point1[1] - point2[1])
    return x+y

def eucDist (point1, point2):
    p = np.math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return p

def getClust(datapoints, centroids):
    clust1=[]
    clust2=[]
    i = 0;
    for d in datapoints:
        i = i+1
        cent = 0
        troid = 0
        for c in centroids:
            dist = manDist(d, c)
            #print("for team ", i," the manhattan distance to centroid: ", c, " is : ", dist)
            #dist = eucDist(d, c)
            #print("for team ", i," the euclidean distance to centroid: ", c, " is : ", dist)
            if cent == 0:
                cent = dist
                troid = 1
            if cent > dist:
                troid = 2
        if troid == 1:
            clust1.append(d)
        else:
            clust2.append(d)
    return clust1, clust2

def mean (points):
    xs=0
    xss=0
    ys=0
    yss=0
    for p in points:
        xs = xs+p[0]
        ys = ys+p[1]
        xss=xss+1
        yss=yss+1
    return ((xs/xss), (ys/yss))
cl1, cl2 = getClust(dp, centr)

########### To run only to find the final centroids ################
while(True):
    cl1, cl2 = getClust(dp, centr)
    mean1=np.round(mean(cl1),2)
    mean2=np.round(mean(cl2),2)
    newCent=[mean1, mean2]
    #print('mean is : ', mean1, ' ', mean2, ' ', mean3)
    if ((centr[0]==newCent[0]).all() and (centr[1]==newCent[1]).all()):
        #print("same")
        break
    centr = [mean1, mean2]

print('clust 1 : ',cl1, ' clust 2 ', cl2)
print('mean is : ', mean(cl1), ' ', mean(cl2))

#cent2 = [mean(cl1), mean(cl2)]
#cl1, cl2 = getClust(dp, cent2)
#print('clust 1 : ',cl1, ' clust 2 ', cl2)
#print('mean 2 is : ', mean(cl1), ' ', mean(cl2))
