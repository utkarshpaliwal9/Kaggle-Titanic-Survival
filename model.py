# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 21:37:16 2018

@author: Utkarsh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir('F:\TItanic Survival')

dataset = pd.read_csv('train.csv')
x = dataset.iloc[:,2:len(dataset.iloc[0,])].values
x = x[:,[0,2,3,4,5,7,9]]
y = dataset.iloc[:,1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:, 1] = labelencoder.fit_transform(x[:, 1])

for i in range(len(x)):
    if x[i,6]=='S':
        x[i,6] = 1
    elif x[i,6]=='C':
        x[i,6] = 2
    elif x[i,6] == 'Q':
        x[i,6] = 3

del(i)

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = "most_frequent")
imputer.fit(x[:,[0,1,3,4,6]])
x[:,[0,1,3,4,6]] = imputer.transform(x[:,[0,1,3,4,6]])
imputer2 = Imputer(strategy = 'mean')
imputer2.fit(x[:,[2,5]])
x[:,[2,5]] = imputer2.transform(x[:,[2,5]])

onehotencoder = OneHotEncoder(categorical_features = [0,6])
x[:,[0,6]] = x[:,[0,6]].astype(np.int64)
x = onehotencoder.fit_transform(x).toarray()
#Column no 6 was unexpected
x = x[:,[1,2,4,5,7,8,9,10]] #Avoiding the Dummy Variable Trap
df_temp2 = pd.DataFrame(x)

#fiting decision tree
from sklearn.linear_model import LogisticRegression
classifier =  LogisticRegression()
classifier.fit(x, y);

#Y-TEST

test_data = pd.read_csv('test.csv')
y_test = test_data.iloc[:,1:len(dataset.iloc[0,])].values
y_test = y_test[:,[0,2,3,4,5,7,9]]

labelencoder = LabelEncoder()
y_test[:, 1] = labelencoder.fit_transform(y_test[:, 1])

df_temp = pd.DataFrame(y_test)

for i in range(len(y_test)):
    if y_test[i,6]=='S':
        y_test[i,6] = 1
    elif y_test[i,6]=='C':
        y_test[i,6] = 2
    elif y_test[i,6] == 'Q':
        y_test[i,6] = 3

del(i)

imputer = Imputer(strategy = "most_frequent")
imputer.fit(y_test[:,[0,1,3,4,6]])
y_test[:,[0,1,3,4, 6]] = imputer.transform(y_test[:,[0,1,3,4,6]])
imputer2 = Imputer(strategy = 'mean')
imputer2.fit(y_test[:,[2,5]])
y_test[:,[2,5]] = imputer2.transform(y_test[:,[2,5]])

onehotencoder = OneHotEncoder(categorical_features = [0,6])
y_test[:,[0,6]] = y_test[:,[0,6]].astype(np.int64)
y_test = onehotencoder.fit_transform(y_test).toarray()
#Column no 6 was unexpected
y_test = y_test[:,[1,2,4,5,7,8,9,10]] #Avoiding the Dummy Variable Trap

#Predicting results
y_pred = classifier.predict(y_test)


df = pd.DataFrame(y_pred)
df.to_csv("F:\TItanic Survival\Predictions.csv")