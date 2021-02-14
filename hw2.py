import pandas as pd
import random
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import pydotplus

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)
#Remove non relevant features
train_df = train_df.drop(['Name', 'SibSp', 'Parch', 'PassengerId', 'Ticket', 'Cabin'], axis = 1)
test_df = test_df.drop(['Name', 'SibSp', 'Parch', 'PassengerId', 'Ticket', 'Cabin'], axis = 1)
#Replace missing values
#print(train_df['Fare'].value_counts())
train_df['Fare'] = train_df[['Fare']].replace(0.0000, 8.0500 )
test_df['Fare'] = test_df[['Fare']].replace(0.0000, 8.0500 )
test_df["Fare"].fillna(8.0500, limit = 1, inplace = True)
mean = train_df['Age'].mean()
standard = train_df['Age'].std()
#print(train_df["Age"])
for (columnName, columnData) in train_df['Age'].iteritems():
    age = random.randint(int(standard),int(mean))
    train_df["Age"].fillna(age, limit = 1, inplace = True)
#convert sex values to numerical
train_df['Gender'] = train_df['Sex']
train_df['Gender'] = train_df[['Gender']].replace('male', 0)
train_df['Gender'] = train_df[['Gender']].replace('female', 1)
train_df['Gender'] = train_df['Gender'].astype(int)
#test
test_df['Gender'] = test_df['Sex']
test_df['Gender'] = test_df[['Gender']].replace('male', 0)
test_df['Gender'] = test_df[['Gender']].replace('female', 1)
test_df['Gender'] = test_df['Gender'].astype(int)
#Remove Sex now that we have gender
train_df = train_df.drop(['Sex'], axis = 1)
test_df = test_df.drop(['Sex'], axis = 1)
#Fill missing Embarked values
train_df["Embarked"].fillna("S", inplace = True)
test_df["Embarked"].fillna("S", inplace = True)
#convert Embarked values to numerical
train_df['Embarked'] = train_df[['Embarked']].replace('S', 0)
train_df['Embarked'] = train_df[['Embarked']].replace('Q', 1)
train_df['Embarked'] = train_df[['Embarked']].replace('C', 2)
train_df['Embarked'] = train_df['Embarked'].astype(int)
#test
test_df['Embarked'] = test_df[['Embarked']].replace('S', 0)
test_df['Embarked'] = test_df[['Embarked']].replace('Q', 1)
test_df['Embarked'] = test_df[['Embarked']].replace('C', 2)
test_df['Embarked'] = test_df['Embarked'].astype(int)

#group up fares
train_df['OFI'] = train_df['Fare'].astype(int)
train_df['OFI'] = np.where(train_df['OFI'].between(-0.001, 7.91), 0, train_df['OFI'])
train_df['OFI'] = np.where(train_df['OFI'].between(7.91, 14.454), 1, train_df['OFI'])
train_df['OFI'] = np.where(train_df['OFI'].between(14.454, 31.0), 2, train_df['OFI'])
train_df['OFI'] = np.where(train_df['OFI'].between(31.0, 512.3299), 3, train_df['OFI'])
#Remove Fare now that we have OFI
train_df = train_df.drop(['Fare'], axis = 1)
#test
#group up fares
test_df['OFI'] = test_df['Fare'].astype(int)
test_df['OFI'] = np.where(test_df['OFI'].between(-0.001, 7.91), 0, test_df['OFI'])
test_df['OFI'] = np.where(test_df['OFI'].between(7.91, 14.454), 1, test_df['OFI'])
test_df['OFI'] = np.where(test_df['OFI'].between(14.454, 31.0), 2, test_df['OFI'])
test_df['OFI'] = np.where(test_df['OFI'].between(31.0, 512.3299), 3, test_df['OFI'])
#Remove Fare now that we have OFI
test_df = test_df.drop(['Fare'], axis = 1)

#group up Age
train_df['AgeG'] = train_df['Age'].astype(int)
train_df['AgeG'] = np.where(train_df['AgeG'].between(-0, 14.5), 0, train_df['AgeG'])
train_df['AgeG'] = np.where(train_df['AgeG'].between(15, 25), 1, train_df['AgeG'])
train_df['AgeG'] = np.where(train_df['AgeG'].between(25.5, 60), 2, train_df['AgeG'])
train_df['AgeG'] = np.where(train_df['AgeG'].between(60, 81), 3, train_df['AgeG'])
#Remove Fare now that we have OFI
train_df = train_df.drop(['Age'], axis = 1)
test_df = test_df.drop(['Age'], axis = 1)

#print(train_df)

#gini index calculation
#Pclass
Pgini1 = train_df.loc[(train_df['Survived'] == 1) & (train_df['Pclass'] == 1)]
Pgini1 = Pgini1['Pclass'].count()
Pgini1N = train_df.loc[(train_df['Survived'] == 0) & (train_df['Pclass'] == 1)]
Pgini1N = Pgini1N['Pclass'].count()
Pgini1T = Pgini1+Pgini1N
giniIndexP1 = 1 - ((Pgini1/Pgini1T)**2 + (Pgini1N/Pgini1T)**2)
#print(giniIndexP1)
Pgini2 = train_df.loc[(train_df['Survived'] == 1) & (train_df['Pclass'] == 2)]
Pgini2 = Pgini2['Pclass'].count()
Pgini2N = train_df.loc[(train_df['Survived'] == 0) & (train_df['Pclass'] == 2)]
Pgini2N = Pgini2N['Pclass'].count()
Pgini2T = Pgini2+Pgini2N
giniIndexP2 = 1 - ((Pgini2/Pgini2T)**2 + (Pgini2N/Pgini2T)**2)
#print(giniIndexP2)
Pgini3 = train_df.loc[(train_df['Survived'] == 1) & (train_df['Pclass'] == 3)]
Pgini3 = Pgini3['Pclass'].count()
Pgini3N= train_df.loc[(train_df['Survived'] == 0) & (train_df['Pclass'] == 3)]
Pgini3N = Pgini3N['Pclass'].count()
Pgini3T = Pgini3+Pgini3N
giniIndexP3 = 1 - ((Pgini3/Pgini3T)**2 + (Pgini3N/Pgini3T)**2)
#print(giniIndexP3)
total = Pgini1T + Pgini2T + Pgini3T
giniIndexP = (Pgini1T/total)*giniIndexP1 + (Pgini2T/total)*giniIndexP2 +(Pgini3T/total)*giniIndexP3
#print("Pclass " , giniIndexP)

#Embarked
Egini1 = train_df.loc[(train_df['Survived'] == 1) & (train_df['Embarked'] == 0)]
Egini1 = Egini1['Embarked'].count()
Egini1N = train_df.loc[(train_df['Survived'] == 0) & (train_df['Embarked'] == 0)]
Egini1N = Egini1N['Embarked'].count()
Egini1T = Egini1+Egini1N
giniIndexE1 = 1 - ((Egini1/Egini1T)**2 + (Egini1N/Egini1T)**2)
#print(giniIndexP1)
Egini2 = train_df.loc[(train_df['Survived'] == 1) & (train_df['Embarked'] == 1)]
Egini2 = Egini2['Embarked'].count()
Egini2N = train_df.loc[(train_df['Survived'] == 0) & (train_df['Embarked'] == 1)]
Egini2N = Egini2N['Embarked'].count()
Egini2T = Egini2+Egini2N
giniIndexE2 = 1 - ((Egini2/Egini2T)**2 + (Egini2N/Egini2T)**2)
#print(giniIndexP2)
Egini3 = train_df.loc[(train_df['Survived'] == 1) & (train_df['Embarked'] == 2)]
Egini3 = Egini3['Embarked'].count()
Egini3N= train_df.loc[(train_df['Survived'] == 0) & (train_df['Embarked'] == 2)]
Egini3N = Egini3N['Embarked'].count()
Egini3T = Egini3+Egini3N
giniIndexE3 = 1 - ((Egini3/Egini3T)**2 + (Egini3N/Egini3T)**2)
#print(giniIndexP3)
totalE = Egini1T + Egini2T + Egini3T
giniIndexE = (Egini1T/totalE)*giniIndexE1 + (Egini2T/totalE)*giniIndexE2 +(Egini3T/totalE)*giniIndexE3
#print("Embarked " ,giniIndexE)

#Gender
Ggini1 = train_df.loc[(train_df['Survived'] == 1) & (train_df['Gender'] == 0)]
Ggini1 = Ggini1['Gender'].count()
Ggini1N = train_df.loc[(train_df['Survived'] == 0) & (train_df['Gender'] == 0)]
Ggini1N = Ggini1N['Gender'].count()
Ggini1T = Ggini1+Ggini1N
giniIndexG1 = 1 - ((Ggini1/Ggini1T)**2 + (Ggini1N/Ggini1T)**2)
#print(giniIndexP1)
Ggini2 = train_df.loc[(train_df['Survived'] == 1) & (train_df['Gender'] == 1)]
Ggini2 = Ggini2['Gender'].count()
Ggini2N = train_df.loc[(train_df['Survived'] == 0) & (train_df['Gender'] == 1)]
Ggini2N = Ggini2N['Gender'].count()
Ggini2T = Ggini2+Ggini2N
giniIndexG2 = 1 - ((Ggini2/Ggini2T)**2 + (Ggini2N/Ggini2T)**2)
#print(giniIndexP3)
totalG = Ggini1T + Ggini2T
giniIndexG = (Ggini1T/totalG)*giniIndexG1 + (Ggini2T/totalG)*giniIndexG2
#print("Gender " ,giniIndexG)

#OFI
Ogini1 = train_df.loc[(train_df['Survived'] == 1) & (train_df['OFI'] == 0)]
Ogini1 = Ogini1['OFI'].count()
Ogini1N = train_df.loc[(train_df['Survived'] == 0) & (train_df['OFI'] == 0)]
Ogini1N = Ogini1N['OFI'].count()
Ogini1T = Ogini1+Ogini1N
giniIndexO1 = 1 - ((Ogini1/Ogini1T)**2 + (Ogini1N/Ogini1T)**2)
Ogini2 = train_df.loc[(train_df['Survived'] == 1) & (train_df['OFI'] == 1)]
Ogini2 = Ogini2['OFI'].count()
Ogini2N = train_df.loc[(train_df['Survived'] == 0) & (train_df['OFI'] == 1)]
Ogini2N = Ogini2N['OFI'].count()
Ogini2T = Ogini2+Ogini2N
giniIndexO2 = 1 - ((Ogini2/Ogini2T)**2 + (Ogini2N/Ogini2T)**2)
Ogini3 = train_df.loc[(train_df['Survived'] == 1) & (train_df['OFI'] == 2)]
Ogini3 = Ogini3['OFI'].count()
Ogini3N= train_df.loc[(train_df['Survived'] == 0) & (train_df['OFI'] == 2)]
Ogini3N = Ogini3N['OFI'].count()
Ogini3T = Ogini3+Ogini3N
giniIndexO3 = 1 - ((Ogini3/Ogini3T)**2 + (Ogini3N/Ogini3T)**2)
Ogini4 = train_df.loc[(train_df['Survived'] == 1) & (train_df['OFI'] == 3)]
Ogini4 = Ogini4['OFI'].count()
Ogini4N= train_df.loc[(train_df['Survived'] == 0) & (train_df['OFI'] == 3)]
Ogini4N = Ogini4N['OFI'].count()
Ogini4T = Ogini4+Ogini4N
giniIndexO4 = 1 - ((Ogini4/Ogini4T)**2 + (Ogini4N/Ogini4T)**2)
totalO = Ogini1T + Ogini2T + Ogini3T + Ogini4T
giniIndexO = (Ogini1T/totalO)*giniIndexO1 + (Ogini2T/totalO)*giniIndexO2 + (Ogini3T/totalO)*giniIndexO3 + (Ogini4T/totalO)*giniIndexO4
#print("OFI " ,giniIndexO)

#AGE
Agini1 = train_df.loc[(train_df['Survived'] == 1) & (train_df['AgeG'] == 0)]
Agini1 = Agini1['AgeG'].count()
Agini1N = train_df.loc[(train_df['Survived'] == 0) & (train_df['AgeG'] == 0)]
Agini1N = Agini1N['AgeG'].count()
Agini1T = Agini1+Agini1N
giniIndexA1 = 1 - ((Agini1/Agini1T)**2 + (Agini1N/Agini1T)**2)
Agini2 = train_df.loc[(train_df['Survived'] == 1) & (train_df['AgeG'] == 1)]
Agini2 = Agini2['AgeG'].count()
Agini2N = train_df.loc[(train_df['Survived'] == 0) & (train_df['AgeG'] == 1)]
Agini2N = Agini2N['AgeG'].count()
Agini2T = Agini2+Agini2N
giniIndexA2 = 1 - ((Agini2/Agini2T)**2 + (Agini2N/Agini2T)**2)
Agini3 = train_df.loc[(train_df['Survived'] == 1) & (train_df['AgeG'] == 2)]
Agini3 = Agini3['AgeG'].count()
Agini3N= train_df.loc[(train_df['Survived'] == 0) & (train_df['AgeG'] == 2)]
Agini3N = Agini3N['AgeG'].count()
Agini3T = Agini3+Agini3N
giniIndexA3 = 1 - ((Agini3/Agini3T)**2 + (Agini3N/Agini3T)**2)
Agini4 = train_df.loc[(train_df['Survived'] == 1) & (train_df['AgeG'] == 3)]
Agini4 = Agini4['AgeG'].count()
Agini4N= train_df.loc[(train_df['Survived'] == 0) & (train_df['AgeG'] == 3)]
Agini4N = Agini4N['AgeG'].count()
Agini4T = Agini4+Agini4N
giniIndexA4 = 1 - ((Agini4/Agini4T)**2 + (Agini4N/Agini4T)**2)
totalA = Agini1T + Agini2T + Agini3T + Agini4T
giniIndexA = (Agini1T/totalA)*giniIndexA1 + (Agini2T/totalA)*giniIndexA2 + (Agini3T/totalA)*giniIndexA3 + (Agini4T/totalA)*giniIndexA4
#print("Age " ,giniIndexA)

#Create tree
train_df = train_df.drop(['AgeG'], axis = 1)
features = ['Pclass', 'Embarked', 'Gender', 'OFI']

X = train_df[features]
y = train_df['Survived']

test_X = test_df[features]

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features)
graph = pydotplus.graph_from_dot_data(data)

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (45,25))
tree.plot_tree(dtree, filled = True, rounded=True, fontsize=10)
#fig.savefig('tree.png')
#plt.show()

#Cross validaion score on decision tree
scores = cross_val_score(dtree, X, y, cv=5)
print("Cross-validation scores on decision tree: {}".format(scores))
print("Average 5-Fold CV Score on decision tree: {}".format(np.mean(scores)))

#Random forest
rf = RandomForestRegressor(n_estimators = 50)
rf.fit(X, y);
#cross validation score on random forest
scores2 = cross_val_score(rf, X, y, cv=5)
print("Cross-validation scores on forest: {}".format(scores2))
print("Average 5-Fold CV Score on forest: {}".format(np.mean(scores2)))

