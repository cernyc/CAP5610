import pandas as pd
import random
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from surprise import Dataset, Reader, KNNBasic, KNNWithMeans
import surprise
from surprise.model_selection import train_test_split, PredefinedKFold
from surprise import BaselineOnly
from surprise.model_selection import cross_validate
from surprise import SVD
from surprise import accuracy
import matplotlib.pyplot as plt

dataset = 'ratings_small.csv'

reader = surprise.Reader(line_format='user item rating timestamp', sep=',',skip_lines=1, rating_scale = (1,5))

data = Dataset.load_from_file(dataset, reader=reader)

trainset, testset = train_test_split(data, test_size=0.25)

# algo = SVD(biased = False)
# results=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
# print("Average rmse of the PMF is ",results['test_rmse'].mean())
# print("Average mae of the PMF is ",results['test_mae'].mean())
#
# algo = KNNBasic()
# results=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
# print("Average rmse of the user collaborative filtering is ",results['test_rmse'].mean())
# print("Average mae of the user collaborative filtering is ",results['test_mae'].mean())
#
# sim_options = {'user_based': False}
# algo = KNNBasic(sim_options=sim_options)
# results=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
# print("Average rmse of the item based collaborative filtering is ",results['test_rmse'].mean())
# print("Average mae of the item based collaborative filtering is ",results['test_mae'].mean())
#
# sim_options = {'name': 'cosine', 'user_based': False}
# algo = KNNBasic(sim_options=sim_options)
# results=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
# cosItem = results['test_rmse'].mean()
# cosItem2 = results['test_mae'].mean()
# print("Average cosine rmse of the item based collaborative filtering is ",cosItem)
# print("Average cosine mae of the item based collaborative filtering is ",cosItem2)
# sim_options = {'name': 'cosine'}
# algo = KNNBasic(sim_options=sim_options)
# results=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
# cosUser = results['test_rmse'].mean()
# cosUser2 = results['test_mae'].mean()
# print("Average cosine rmse of the user based collaborative filtering is ",cosUser)
# print("Average cosine mae of the user based collaborative filtering is ",cosUser2)
#
# sim_options = {'name': 'msd', 'user_based': False}
# algo = KNNBasic(sim_options=sim_options)
# results=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
# msdItem = results['test_rmse'].mean()
# msdItem2 = results['test_mae'].mean()
# print("Average msd rmse of the item based collaborative filtering is ",msdItem)
# print("Average msd mae of the item based collaborative filtering is ",msdItem2)
# sim_options = {'name': 'msd'}
# algo = KNNBasic(sim_options=sim_options)
# results=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
# msdUser = results['test_rmse'].mean()
# msdUser2 = results['test_mae'].mean()
# print("Average msd rmse of the user based collaborative filtering is ",msdUser)
# print("Average msd mae of the user based collaborative filtering is ",msdUser2)
#
# sim_options = {'name': 'pearson', 'user_based': False}
# algo = KNNBasic(sim_options=sim_options)
# results=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
# pearItem = results['test_rmse'].mean()
# pearItem2 = results['test_mae'].mean()
# print("Average pearson rmse of the item based collaborative filtering is ",pearItem)
# print("Average pearson mae of the item based collaborative filtering is ",pearItem2)
# sim_options = {'name': 'pearson'}
# algo = KNNBasic(sim_options=sim_options)
# results=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
# pearUser = results['test_rmse'].mean()
# pearUser2 = results['test_mae'].mean()
# print("Average pearson rmse of the user based collaborative filtering is ",pearUser)
# print("Average pearson mae of the user based collaborative filtering is ",pearUser2)
#
# x=['cosine','msd','pearson']
# y=[cosItem,msdItem,pearItem]
# y2=[cosItem2,msdItem2,pearItem2]
# y3=[cosUser,msdUser,pearUser]
# y4=[cosUser2,msdUser2,pearUser2]
# plt.plot(x, y, 'o', color='black')
# plt.plot(x, y2, 'o', color='black')
# plt.plot(x, y3, 's', color='red')
# plt.plot(x, y4, 's', color='red')
# plt.show()

my_k = 10
sim_option = {'user_based':False,}
#algo = KNNWithMeans(k = my_k,  sim_option = sim_option)
algo = KNNWithMeans(k = my_k)
results10=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
print("Average rmse with k=10 is ",results10['test_rmse'].mean())
print("Average mae with k-=10 is ",results10['test_mae'].mean())

my_k = 15
sim_option = {'user_based':False,}
#algo = KNNWithMeans(k = my_k,  sim_option = sim_option)
algo = KNNWithMeans(k = my_k)
results15=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
print("Average rmse with k=15 is ",results15['test_rmse'].mean())
print("Average mae with k=15 is ",results15['test_mae'].mean())

my_k = 20
sim_option = {'user_based':False,}
#algo = KNNWithMeans(k = my_k,  sim_option = sim_option)
algo = KNNWithMeans(k = my_k)
results20=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
print("Average rmse with k=20 is ",results20['test_rmse'].mean())
print("Average mae with k=20 is ",results20['test_mae'].mean())

my_k = 25
sim_option = {'user_based':False,}
#algo = KNNWithMeans(k = my_k,  sim_option = sim_option)
algo = KNNWithMeans(k = my_k)
results25=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
print("Average rmse with k=25 is ",results25['test_rmse'].mean())
print("Average mae with k=25 is ",results25['test_mae'].mean())

my_k = 35
sim_option = {'user_based':False,}
#algo = KNNWithMeans(k = my_k,  sim_option = sim_option)
algo = KNNWithMeans(k = my_k)
results35=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
print("Average rmse with k=35 is ",results35['test_rmse'].mean())
print("Average mae with k=35 is ",results35['test_mae'].mean())

my_k = 50
sim_option = {'user_based':False,}
#algo = KNNWithMeans(k = my_k,  sim_option = sim_option)
algo = KNNWithMeans(k = my_k)
results50=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
print("Average rmse with k=50 is ",results50['test_rmse'].mean())
print("Average mae with k=50 is ",results50['test_mae'].mean())

my_k = 75
sim_option = {'user_based':False,}
#algo = KNNWithMeans(k = my_k,  sim_option = sim_option)
algo = KNNWithMeans(k = my_k)
results75=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
print("Average rmse with k=75 is ",results75['test_rmse'].mean())
print("Average mae with k=75 is ",results75['test_mae'].mean())

my_k = 100
sim_option = {'user_based':False,}
#algo = KNNWithMeans(k = my_k,  sim_option = sim_option)
algo = KNNWithMeans(k = my_k)
results100=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
print("Average rmse with k=100 is ",results100['test_rmse'].mean())
print("Average mae with k=100 is ",results100['test_mae'].mean())

my_k = 150
sim_option = {'user_based':False,}
#algo = KNNWithMeans(k = my_k,  sim_option = sim_option)
algo = KNNWithMeans(k = my_k)
results150=cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, return_train_measures=True)
print("Average rmse with k=150 is ",results150['test_rmse'].mean())
print("Average mae with k=150 is ",results150['test_mae'].mean())

x=['10', '15', '20', '25','35','50','75','100','150']
y=[results10['test_rmse'].mean(),results15['test_rmse'].mean(),results20['test_rmse'].mean(),results25['test_rmse'].mean(),results35['test_rmse'].mean(),results50['test_rmse'].mean(),results75['test_rmse'].mean(),results100['test_rmse'].mean(),results150['test_rmse'].mean()]
y2=[results10['test_mae'].mean(),results15['test_mae'].mean(),results20['test_mae'].mean(),results25['test_mae'].mean(),results35['test_mae'].mean(),results50['test_mae'].mean(),results75['test_mae'].mean(),results100['test_mae'].mean(),results150['test_mae'].mean()]
plt.plot(x, y, 'o', color='black')
plt.show()
plt.plot(x, y2, 's', color='red')
plt.show()

