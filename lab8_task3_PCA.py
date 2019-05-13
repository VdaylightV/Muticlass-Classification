# -*- coding: utf-8 -*-
"""
Created on Mon May 13 21:12:16 2019

@author: Administrator
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS

matrix = []
type_lst = []
type_origin=[]
filename = 'E:/大学课程/AI程序设计/实验8降维回归和分类/dataset.csv'

data = pd.read_csv(filename,engine = 'python')
DF = data.drop('A11', axis = 1)
DF_matrix_origin = DF.values
for i in range(0,1500):
    if DF_matrix_origin[i][0]=='A':
        type_lst.append(1)
    elif DF_matrix_origin[i][0]=='B':
        type_lst.append(2)
    elif DF_matrix_origin[i][0]=='C':
        type_lst.append(3)
    else:
        type_lst.append(4)
DF_2 = DF.drop('class', axis = 1)
DF_matrix = DF_2.values

X = np.mat(DF_matrix)  
Y = np.array(DF_matrix)
#print(type_lst)

##################利用PCA把数据降维，降成3维
pca = PCA(n_components = 3)
reduced_X = pca.fit_transform(Y)
#print(reduced_X)
reduced_X_matrix = np.mat(reduced_X)
#print(reduced_X)
#
#################形成列表，为后面哦数据可视化作准备
X_axis_1 = []
Y_axis_1 = []
Z_axis_1 = []

X_axis_2 = []
Y_axis_2 = []
Z_axis_2 = []

X_axis_3 = []
Y_axis_3 = []
Z_axis_3 = []

X_axis_4 = []
Y_axis_4 = []
Z_axis_4 = []

X_axis = []
Y_axis = []
Z_axis = []
for i in range(0,1500):
    if type_lst[i]==1:
        X_axis_1.append(reduced_X[i][0])
        Y_axis_1.append(reduced_X[i][1])
        Z_axis_1.append(reduced_X[i][2])
    elif type_lst[i]==2:
        X_axis_2.append(reduced_X[i][0])
        Y_axis_2.append(reduced_X[i][1])
        Z_axis_2.append(reduced_X[i][2])
    elif type_lst[i]==3:
        X_axis_3.append(reduced_X[i][0])
        Y_axis_3.append(reduced_X[i][1])
        Z_axis_3.append(reduced_X[i][2])
    else:
        X_axis_4.append(reduced_X[i][0])
        Y_axis_4.append(reduced_X[i][1])
        Z_axis_4.append(reduced_X[i][2])
    X_axis.append(reduced_X[i][0])
    Y_axis.append(reduced_X[i][1])
    Z_axis.append(reduced_X[i][2])

##################进行数据的可视化
ax = plt.figure().add_subplot(111, projection = '3d')
ax.scatter(X_axis_1,Y_axis_1,Z_axis_1, c='b',marker='o')
ax.scatter(X_axis_2,Y_axis_2,Z_axis_2, c='r',marker='D')
ax.scatter(X_axis_3,Y_axis_3,Z_axis_3, c='g',marker='*')
ax.scatter(X_axis_4,Y_axis_4,Z_axis_4, c='y',marker='s')

ax.set_xlabel('X Label_PCA')
ax.set_ylabel('Y Labe_PCA')
ax.set_zlabel('Z Label_PCA')

plt.show()