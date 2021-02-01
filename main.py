import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]
pd.set_option('display.max_columns', None)
pd.set_option('display.max_row', None)
#print (train_df)
#################################################
#Q1:
#print (train_df.keys())
#['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
#################################################
#Q2:
#Survived, Sex, Age, Pclass, SibSp, Parch, Fare, Embarked  --- total 8
#################################################
#Q3:
#print(train_df.dtypes)
num_df = train_df [['Parch', 'Age', 'SibSp', 'Fare']]
#PassengerId, Survived, Pclass, Parch, Age, SibSp, Fare -- total 7
#################################################
#Q4:
#print(train_df.dtypes)
#Ticket, Cabin
#################################################
#Q5:
#print(train_df.isna().sum())
# train set : Age, Cabin, Embarked -- total 3
cat_df = train_df[['PassengerId', 'Survived', 'Sex', 'Pclass', 'Ticket', 'Cabin', 'Name', 'Embarked']]
#print(test_df.isna().sum())
# test set : Age, Fare, Cabin
#################################################
#Q6:
#print(train_df.dtypes)
#data types are the following:
# PassengerId int64, Survived int64, Pclass int64, Name object, Sex object, Age float64, SibSp int64
# Parch int64, Ticket object, Fare float64, Cabin object, Embarked object.
#################################################
#Q7
#print(num_df.mean())
#print(num_df.count())
#print(num_df.std())
#print(num_df.quantile(.25))
#print(num_df.quantile(.5))
#print(num_df.quantile(.75))
#print(num_df.min())
#print(num_df.max())
#################################################
#Q8:
#cat_df = train_df[['PassengerId', 'Survived', 'Sex', 'Pclass', 'Ticket', 'Cabin', 'Name', 'Embarked']]
#print(cat_df[['PassengerId']].count())
#print(cat_df[['PassengerId']].value_counts())
#print(cat_df[['PassengerId']].value_counts().count())
#print(cat_df[['Survived']].count())
#print(cat_df[['Survived']].value_counts())
#print(cat_df[['Survived']].value_counts().count())
#print(cat_df[['Sex']].count())
#print(cat_df[['Sex']].value_counts())
#print(cat_df[['Sex']].value_counts().count())
#print(cat_df[['Pclass']].count())
#print(cat_df[['Pclass']].value_counts())
#print(cat_df[['Pclass']].value_counts().count())
#print(cat_df[['Ticket']].count())
#print(cat_df[['Ticket']].value_counts())
#print(cat_df[['Ticket']].value_counts().count())
#print(cat_df[['Cabin']].value_counts())
#print(cat_df[['Cabin']].value_counts().count())
#print(cat_df[['Cabin']].count())
#print(cat_df[['Name']].value_counts())
#print(cat_df[['Name']].value_counts().count())
#print(cat_df[['Name']].count())
#print(cat_df[['Embarked']].value_counts())
#print(cat_df[['Embarked']].value_counts().count())
#print(cat_df[['Embarked']].count())
#################################################
#Q9:
#class1Sur = train_df.loc[(train_df['Pclass'] == 1) & (train_df['Survived'] == 1)]
#class1NSur = train_df.loc[(train_df['Pclass'] == 1) & (train_df['Survived'] == 0)]
#print(class1Sur.shape[0])
#print(class1NSur.shape[0]+class1Sur.shape[0])
#print( 1 - (class1NSur.shape[0]/class1Sur.shape[0]))
#class2Sur = train_df.loc[(train_df['Pclass'] == 2) & (train_df['Survived'] == 1)]
#class2NSur = train_df.loc[(train_df['Pclass'] == 2) & (train_df['Survived'] == 0)]
#print(class2Sur.shape[0])
#print(class2NSur.shape[0]+class2Sur.shape[0])
#class3Sur = train_df.loc[(train_df['Pclass'] == 3) & (train_df['Survived'] == 1)]
#class3NSur = train_df.loc[(train_df['Pclass'] == 3) & (train_df['Survived'] == 0)]
#print(class3Sur.shape[0])
#print(class3NSur.shape[0]+class3Sur.shape[0])
#################################################
#Q10
#WomSur = train_df.loc[(train_df['Sex'] == "female") & (train_df['Survived'] == 1)]
#WomNSur = train_df.loc[(train_df['Sex'] == "female") & (train_df['Survived'] == 0)]
#print(WomSur.shape[0])
#print(WomNSur.shape[0])
#print(WomSur.shape[0]/(WomNSur.shape[0]+WomSur.shape[0]))
#MenSur = train_df.loc[(train_df['Sex'] == "male") & (train_df['Survived'] == 1)]
#MenNSur = train_df.loc[(train_df['Sex'] == "male") & (train_df['Survived'] == 0)]
#print(MenSur.shape[0]/(MenNSur.shape[0]+MenSur.shape[0]))
#################################################
#Q11
#inSurv = train_df.loc[(train_df['Survived'] == 1) & (train_df['Age'] <= 4)]
#inNSurv = train_df.loc[(train_df['Survived'] == 0) & (train_df['Age'] <= 4)]
#print(inSurv.count())
#print(inNSurv.count())
#oldSurv = train_df.loc[(train_df['Survived'] == 1) & (train_df['Age'] == 80)]
#oldNSurv = train_df.loc[(train_df['Survived'] == 0) & (train_df['Age'] == 80)]
#print(oldSurv.count())
#print(oldNSurv.count())
#midSurv = train_df.loc[(train_df['Survived'] == 1) & (train_df['Age'] >= 15) & (train_df['Age'] <= 25)]
#midNSurv = train_df.loc[(train_df['Survived'] == 0) & (train_df['Age'] >= 15) & (train_df['Age'] <= 25)]
#print(midSurv.count())
#print(midNSurv.count())
#Surv = train_df.loc[(train_df['Survived'] == 1)]
#NSurv = train_df.loc[(train_df['Survived'] == 0)]
#Surv['Age'].plot.hist(bins = 80)
#plt.show()
#NSurv['Age'].plot.hist(bins = 80)
#plt.show()
#################################################
#Q12:
#P1Surv = train_df.loc[(train_df['Survived'] == 1) & (train_df['Pclass'] == 1)]
#P1NSurv = train_df.loc[(train_df['Survived'] == 0) & (train_df['Pclass'] == 1)]
#P2Surv = train_df.loc[(train_df['Survived'] == 1) & (train_df['Pclass'] == 2)]
#P2NSurv = train_df.loc[(train_df['Survived'] == 0) & (train_df['Pclass'] == 2)]
#P3Surv = train_df.loc[(train_df['Survived'] == 1) & (train_df['Pclass'] == 3)]
#P3NSurv = train_df.loc[(train_df['Survived'] == 0) & (train_df['Pclass'] == 3)]
#P1Surv['Age'].plot.hist(bins = 80)
#plt.show()
#P1NSurv['Age'].plot.hist(bins = 80)
#plt.show()
#P2Surv['Age'].plot.hist(bins = 80)
#plt.show()
#P2NSurv['Age'].plot.hist(bins = 80)
#plt.show()
#P3Surv['Age'].plot.hist(bins = 80)
#plt.show()
#P3NSurv['Age'].plot.hist(bins = 80)
#plt.show()
#################################################
#Q13
#ESSurv = train_df.loc[(train_df['Survived'] == 1) & (train_df['Embarked'] == "S")]
#sns.factorplot(x = ESSurv['Sex'], y = ESSurv['Fare'], data = ESSurv)
#print(ESSurv.head())
#plt.show()
#ESNSurv = train_df.loc[(train_df['Survived'] == 0) & (train_df['Embarked'] == "S")]
#sns.factorplot(x = ESNSurv['Sex'], y = ESNSurv['Fare'], data = ESNSurv)
#print(ESNSurv.head())
#plt.show()
#ECSurv = train_df.loc[(train_df['Survived'] == 1) & (train_df['Embarked'] == "C")]
#sns.factorplot(x = ECSurv['Sex'], y = ECSurv['Fare'], data = ECSurv)
#print(ECSurv.head())
#plt.show()
#ECSNurv = train_df.loc[(train_df['Survived'] == 0) & (train_df['Embarked'] == "C")]
#sns.factorplot(x = ECSNurv['Sex'], y = ECSNurv['Fare'], data = ECSNurv)
#print(ECSNurv.head())
#plt.show()
#EQSurv = train_df.loc[(train_df['Survived'] == 1) & (train_df['Embarked'] == "Q")]
#sns.factorplot(x = EQSurv['Sex'], y = EQSurv['Fare'], data = EQSurv)
#print(EQSurv.head())
#plt.show()
#EQNSurv = train_df.loc[(train_df['Survived'] == 0) & (train_df['Embarked'] == "Q")]
#sns.factorplot(x = EQNSurv['Sex'], y = EQNSurv['Fare'], data = EQNSurv)
#print(EQNSurv.head())
#plt.show()
#################################################
#Q14
#FDuplicate = train_df
#print(FDuplicate['Ticket'].count())
#DTicket = train_df.pivot_table(index = ['Ticket'], aggfunc ='size')
#FDuplicate["dup"] = FDuplicate.duplicated(subset='Ticket', keep='first')
#print(FDuplicate)
#FDuplicate = FDuplicate.loc[(FDuplicate['dup'] == True)]
#FNDuplicate = FDuplicate.loc[(FDuplicate['dup'] == False)]
#print(FDuplicate['dup'].count())
#FDuplicateS = FDuplicate.loc[(FDuplicate['Survived'] == 1)]
#print(FDuplicateS['dup'].count())
#FFDuplicateS = FNDuplicate.loc[(FNDuplicate['Survived'] == 1)]
#print(FFDuplicateS['dup'].count())
#################################################
#Q15
#NCab = train_df
#NTCab = test_df
#print(NCab['Cabin'].isnull().sum())
#print(NCab['Cabin'].isnull().sum() + NTCab['Cabin'].isnull().sum())
#################################################
#Q16
#ToNum = train_df
#ToNum['Gender'] = ToNum['Sex']
#ToNum['Gender'] = ToNum[['Gender']].replace('male', 0)
#ToNum['Gender'] = ToNum[['Gender']].replace('female', 1)
#ToNum['Gender'] = ToNum['Gender'].astype(int)
#print(ToNum)
#################################################
#Q17
#mean = train_df['Age'].mean()
#standard = train_df['Age'].std()
#print(train_df["Age"])
#for (columnName, columnData) in train_df['Age'].iteritems():
#    age = random.randint(int(standard),int(mean))
#    train_df["Age"].fillna(age, limit = 1, inplace = True)
#print(train_df["Age"])
#################################################
#Q18
#print(train_df['Embarked'])
#print(train_df['Embarked'].value_counts())
#most common is S
#train_df["Embarked"].fillna("S", inplace = True)
#print(train_df['Embarked'])
#################################################
#Q19
#print(train_df['Fare'])
#print(train_df['Fare'].value_counts())
#test_df['Fare'] = test_df[['Fare']].replace(0.0000, 8.0500 )
#print(test_df['Fare'].value_counts())
#################################################
#Q20
#train_df['OFI'] = train_df['Fare'].astype(int)
#train_df['OFI'] = np.where(train_df['OFI'].between(-0.001, 7.91), 0, train_df['OFI'])
#train_df['OFI'] = np.where(train_df['OFI'].between(7.91, 14.454), 1, train_df['OFI'])
#train_df['OFI'] = np.where(train_df['OFI'].between(14.454, 31.0), 2, train_df['OFI'])
#train_df['OFI'] = np.where(train_df['OFI'].between(31.0, 512.3299), 3, train_df['OFI'])
#print(train_df['OFI'])
