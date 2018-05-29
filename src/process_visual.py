# -*- coding: utf-8 -*-
"""
Created on Thu May 04 15:11:13 2017

@author: Jian Yang local
"""

import csv
import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
 

my_data = pd.read_csv('train_advhousing.csv')

print "dimension before clean", my_data.shape #1460, 81
all_features = list(my_data.columns.values)

#separate numeric features from category features 
categorical_features = ['MSSubClass','MSZoning', 'Street','LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1','Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',  'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond', 'PavedDrive',  'SaleType', 'SaleCondition', 'MiscFeature','Fence','Alley','PoolQC']
numeric_features = list(set(all_features)- set(categorical_features)) #26
#print all_features, len(all_features)       
print numeric_features, len(numeric_features)       #26 numeric features 
print categorical_features, len(categorical_features)  # 55 category features


"""
Impute numeric features
"""
# for GarageYrBlt missing, fill with yearbuilt
garageYr=my_data['GarageYrBlt'].fillna(my_data['YearBuilt'])

#for others, just fill in median value
imp = pp.Imputer(missing_values='NaN', strategy='median')
imp = imp.fit(my_data[numeric_features])     
my_data[numeric_features]=imp.transform(my_data[numeric_features])


#B= np.array(my_data['GarageYrBlt'])
#print np.count_nonzero(B == 0)
"""
Impute categorical features
"""
#for categorical_features, remove those columns mostly "NA",
#while for other columns replace "NA" with "ZZ"
to_remove = []
for item in categorical_features:
    n = my_data[item].isnull().sum()
    if n>1000:
       to_remove.append(item)
       print item + ":" , n
    else:
       my_data[item]= my_data[item].fillna("ZZ")

"""
#delete unnecessary columns for machine learning and data visualization
"""           
del my_data["Id"]           
for item in to_remove:
    del my_data[item]

"""
save data for data visualization
"""
print "dimension after cleaning", my_data.shape
my_data.to_csv("hdataforVisua.csv", index = False)        
#       

"""
Encoding only categorical variables for machine learning
"""
le=pp.LabelEncoder()

for col in my_data.columns.values:
    if my_data[col].dtypes=='object':
    # Using whole data to form an exhaustive list of levels
        le.fit(my_data[col].values)
        my_data[col]=le.transform(my_data[col])
        
#a = list(le.classes_)
#print le.transform(a)
"""
Save data for machine learning
"""
my_data.to_csv("processedforML.csv", index = False)   
          
#np.savetxt('test.csv', my_data, delimiter=',')