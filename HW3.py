import pandas as pd
import random
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
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

train_df['Pclass'] = train_df['Pclass'].astype(str)
#print(train_df["Pclass"])

#Remove Fare now that we have OFI
train_df = train_df.drop(['Age'], axis = 1)
test_df = test_df.drop(['Age'], axis = 1)
#print(train_df)

features = ['Pclass', 'Embarked', 'Gender', 'OFI']
X = train_df[features]
y = train_df['Survived']

clf = svm.SVC(kernel='linear', C=1).fit(X, y)
scores = cross_val_score(clf, X, y, cv=5)
print('score for linear is: ', scores)
print("Average 5-Fold CV Score for Linear SVM is : {}".format(np.mean(scores)))

clfp = svm.SVC(kernel='poly', C=1).fit(X, y)
scoresp = cross_val_score(clfp, X, y, cv=5)
print('score for quadratic is: ', scoresp)
print("Average 5-Fold CV Score for quadratic SVM is : {}".format(np.mean(scoresp)))

clfr = svm.SVC(kernel='rbf', C=1).fit(X, y)
scoresr = cross_val_score(clfr, X, y, cv=5)
print('score for rbf is: ', scoresr)
print("Average 5-Fold CV Score for RBF SVM is : {}".format(np.mean(scoresr)))