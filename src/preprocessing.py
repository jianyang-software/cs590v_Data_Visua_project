# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 00:57:30 2017

@author: Jian Yang local
"""
import csv
import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
 

my_data = pd.read_csv('train_advhousing.csv')
print list(my_data.columns.values)
numeric_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','TotalBsmtSF','Fireplaces', 'GarageCars', 'GarageArea','WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

# Impute Only impute for numeric variables, categorical will have a NA column
imp = pp.Imputer(missing_values='NaN', strategy='median')
imp = imp.fit(my_data[numeric_features])     
my_data[numeric_features]=imp.transform(my_data[numeric_features])


# for GarageYrBlt missing, fill with yearbuilt
garageYr=my_data['GarageYrBlt'].fillna(my_data['YearBuilt'])

my_data['GarageYrBlt']=garageYr
# encoding only categorical variables
le=pp.LabelEncoder()

for col in my_data.columns.values:
    # Encoding only categorical variables
    if my_data[col].dtypes=='object':
    # Using whole data to form an exhaustive list of levels
        data = my_data[col]
        le.fit(data.values)
        my_data[col]=le.transform(my_data[col])
        
        
#a = list(le.classes_)
#print le.transform(a)


my_data.to_csv("hdataforML.csv", index = False)           
##np.savetxt('test.csv', my_data, delimiter=',')