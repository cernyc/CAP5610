import random
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

############# IRIS DATA IMPORT ################################
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
df = pd.read_csv("iris.data", names=colnames)
dd = df.values
X = df.values[:, 0:4]
y = df.values[:, 4]
sse=0

############# GENERATE RANDOM INITIAL CENTROIDS #####################
centr = [[random.uniform(df['sepal-length'].min(),df['sepal-length'].max()), random.uniform(df['sepal-width'].min(),df['sepal-width'].max()), random.uniform(df['petal-length'].min(),df['petal-length'].max()), random.uniform(df['petal-width'].min(),df['petal-width'].max())],
         [random.uniform(df['sepal-length'].min(),df['sepal-length'].max()), random.uniform(df['sepal-width'].min(),df['sepal-width'].max()), random.uniform(df['petal-length'].min(),df['petal-length'].max()), random.uniform(df['petal-width'].min(),df['petal-width'].max())],
         [random.uniform(df['sepal-length'].min(),df['sepal-length'].max()), random.uniform(df['sepal-width'].min(),df['sepal-width'].max()), random.uniform(df['petal-length'].min(),df['petal-length'].max()), random.uniform(df['petal-width'].min(),df['petal-width'].max())]]

############# CALCULATE MANHATTAN DISTANCE BETWEEN TWO POINTS ##############
def manDist (point1, point2):
    temp=0
    for i in range(len(point2)-1):
        if point1[i] < point2[i]:
            x = (point2[i] - point1[i])
        else:
            x = (point1[i] - point2[i])
        temp = temp+x
    return temp
############### CALCULATE EUCLEDIAN DISTANCE BETWEEN TWO POINTS ##################
def eucDist (point1, point2):
    temp = 0
    for i in range(len(point2)-1):
        temp = temp + ((point1[i] - point2[i])**2)
    p = np.math.sqrt(temp)
    return p

################ CALCULATE COSINE DISTANCE BETWEEN TWO POINTS ###################
def cosDist (point1, point2):
    format=np.delete(point1, 4, 0)
    mult = sum(p1*p2 for p1, p2 in zip(format, point2))
    norm_a = sum(p1*p1 for p1 in format) ** 0.5
    norm_b = sum(p2*p2 for p2 in point2) ** 0.5
    #result = dot(format, point2)/(norm(format)*norm(point2))
    #result = 1 - spatial.distance.cosine(format, point2)
    result = mult / (norm_a*norm_b)
    return 1-(result)

############### CALCULATE JARCARD DISTANCE BETWEEN TWO POINTS #####################
def jacDist(list1, list2):
    inter = len(list(set(list1).intersection(list2)))
    return 1-(float(inter) / ((len(list1) + len(list2)) - inter))

############## GIVEN THE POINTS AND THE CENTROIDS GET THE CLUSTERS ###############
def getClust(datapoints, centroids):
    clust1=[]
    clust2=[]
    clust3=[]
    i=0
    sse=0
    for d in datapoints:
        i = i+1
        cent = 0
        troid = 0
        clus = 0
        for c in centroids:
            clus = clus+1
############# UNCOMMENT THE METRICS TO BE USED ####################
            #dist = manDist(d, c)
            dist = eucDist(d, c)
            #dist = cosDist(d, c)
            #dist = jacDist(d, c)
            sse=sse+(dist**2)
            if cent == 0:
                cent = dist
                troid = 1
            elif cent > dist:
                troid = clus
                cent = dist
        if troid == 1:
            clust1.append(d)
        if troid == 2:
            clust2.append(d)
        if troid == 3:
            clust3.append(d)
    return clust1, clust2, clust3, sse

############# GET THE KMEAN #############
def mean (points):
    s1=0
    s2=0
    s3=0
    s4=0
    v=0
    for p in points:
        s1 = s1+p[0]
        s2 = s2+p[1]
        s3 = s3+p[2]
        s4 = s4+p[3]
        v=v+1
    if v>0:
        return ((s1/v), (s2/v), (s3/v), (s4/v))
    else:
        return [random.uniform(df['sepal-length'].min(),df['sepal-length'].max()), random.uniform(df['sepal-width'].min(),df['sepal-width'].max()), random.uniform(df['petal-length'].min(),df['petal-length'].max()), random.uniform(df['petal-width'].min(),df['petal-width'].max())]

########## METHOD TO GET THE CLUSTER ACCURACY WITH DATA FIELD ###############
def getError(cluster):
    ise=0
    ive=0
    ivi=0
    for z in cluster:
        #print(z[4])
        if (z[4] == 'Iris-setosa'):
            ise=ise+1
        if (z[4] == 'Iris-versicolor'):
            ive=ive+1
        if (z[4] == 'Iris-virginica'):
            ivi=ivi+1
    if (ise > ive):
        if (ise > ivi):
            return (ive+ivi)
        else:
            return (ive+ise)
    if (ivi > ive):
        return ive+ise
    else:
        return ivi+ise

################ GET SSE ###################
def getSSE(cluster, centroid):
    squared_errors=0
    for z in cluster:
        format = np.delete(z, 4, 0)
        squared_errors = squared_errors + ((centroid - format) ** 2)
    return squared_errors

##################################################
#############  MAIN CODE  ########################
##################################################

########## GET THE INITIAL CLUSTERS #############
cl1, cl2, cl3,sse = getClust(dd, centr)
x=0
prevsse=sse

while(True):
    cl1, cl2, cl3, sse = getClust(dd, centr)
    #print('clust 1 : ',len(cl1), ' clust 2 ', len(cl2), ' clust 3 ', len(cl3))
    mean1=np.round(mean(cl1),2)
    mean2=np.round(mean(cl2),2)
    mean3=np.round(mean(cl3),2)
    newCent=[mean1, mean2, mean3]
    #print('mean is : ', mean1, ' ', mean2, ' ', mean3)
    if ((centr[0]==newCent[0]).all() and (centr[1]==newCent[1]).all() and (centr[2]==newCent[2]).all()):
        #print("same")
        break
##############Q4 CONDITION, UNCOMMENT TO TEST ##########
    #if(x>1000):
    #   break
    #if sse>prevsse:
    #    break
    #else:
    #    prevsse=sse
    centr = [mean1, mean2, mean3]
    x=x+1

print('clust 1 : ',len(cl1), ' clust 2 ', len(cl2), ' clust 3 ', len(cl3))
print('Centroids are : ', mean(cl1), ' ', mean(cl2), ' ', mean(cl3))

a=getError(cl1)
b=getError(cl2)
c=getError(cl3)
########## GET ACCURACY #########
err = (150-(a+b+c))/150
print('Accuracy is : ', err)
########## GET SSE #########
sse=getSSE(cl1, centr[0])
print('SSE is : ', sse)

########### UNCOMMENT TO GET NUMBER OF ITERATION ############
print('Number of iteration is : ',x)



################  TASK 3  #################
########### UNCOMMENT TO TEST #############

#print(eucDist([4.7, 3.2],[6.2, 2.8]))
#print(eucDist([4.6, 2.9],[6.7, 3.1]))
#print(eucDist([4.6, 2.9],[5.9, 3.2]))
#print(eucDist([5.0, 3.0],[5.9, 3.2]))
#print(eucDist([4.9, 3.1],[5.9, 3.2]))
#red = [[4.7, 3.2], [4.9,3.1], [5.0,3.0], [4.6,2.9]]
#blue = [[5.9,3.2], [6.7, 3.1], [6.0, 3.0], [6.2, 2.8]]

#it = 0
#ave = 0
#for l in range(len(red)):
#    for d in range(len(blue)):
#        ave = ave + eucDist(red[l],blue[d])
#        it=it+1
#print('average ', ave/it)