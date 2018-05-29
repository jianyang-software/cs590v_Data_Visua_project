# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:58:43 2017

@author: Jian Yang local
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression



def plot_error_chart(hypername, hyperP_list, train_err, valid_err):
    """
    plot training and validation error in cross validation 
    """
   
    plt.plot(hyperP_list, train_err, 'r-', linewidth=4)
    plt.plot(hyperP_list, valid_err, 'b-', linewidth=4)
    plt.grid(True)
    plt.xlabel(hypername)
    plt.ylabel("Error Rate")
    plt.title("Train-Test Error Curves "+ hypername)
    leg = plt.legend(["Train Err","Valid Err"]);
    leg.get_frame().set_alpha(0.5)
    plt.show()
    
    
def get_fea_importance(selectF,featurename, train_x, train_y):
    """
    Generate csv file with features and their score, for data visualization
    """
    
    tr_x = train_x
    tr_y = train_y
    
    # feature selection
    selector = selectF.fit(tr_x, tr_y)
    sel_fea_idx= selector.get_support(True)
        
    # arrange features and their scores in decending order 
    order_score = np.sort(selector.scores_)[::-1]
    order_index = np.argsort(selector.scores_)[::-1]
    
    # plot rank of feature in decending order 
    plt.Figure()
    plt.plot(sel_fea_idx, order_score,'r-+', linewidth =4)
    
    data = np.array([featurename[order_index], order_score]).transpose()
    print data.shape
    
    # save features and their corresponding score to csv for data-visualization
    df = pd.DataFrame(data) 
    df.to_csv("featureImportance.csv", header = ["Feature", "Score"], index =False)
    

    
def pipLine(train_x, train_y, test_x, regressor, paragrid, selectF):
    """
    Th is is the pipline function, it taks 3 parameters and datasets:
        1. regressor()
        2. paragrid: hyperparameters for cross-validation
        3. a method of selecting features: selectF
    """
    tr_x = train_x
    tr_y = train_y
    
    tr_score = []
    vl_score = []
   
       
    # feature selection
    selector = selectF.fit(tr_x, tr_y)
    tr_x_new = selector.transform(tr_x)
    
    sel_fea_idx= selector.get_support(True)
    
   
#    print tr_x_new.shape, selector.scores_,sel_fea_idx
#    print feature_name[sel_fea_idx]
    
    #grid_search with cross validation   
    #score = "R2 metric"  
    score = "r2"              
    estimator = GridSearchCV(regressor, cv=5,
                       param_grid=paragrid, scoring =score)
    estimator.fit (tr_x_new, tr_y )
    print("Best parameters set found on development set:")
    print()
    print(estimator.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    
    means = estimator.cv_results_['mean_test_score']
    stds = estimator.cv_results_['std_test_score']
    
    for mean, std, params in zip(means, stds, estimator.cv_results_['params']):
        
        print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std*2 , params))
    print()
        
        
    print("Detailed regression report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    
    tr_score = estimator.cv_results_["mean_train_score"]
    vl_score = estimator.cv_results_['mean_test_score']

    y_pred = estimator.predict(test_x[:,sel_fea_idx])
    return y_pred,tr_score, vl_score


my_data = np.genfromtxt('processedforML.csv', delimiter=',')

# the 1st row is supposed to be colname, don't need for model; the 1st column is the id, not 
#needed for model
M_data = pd.read_csv('processedforML.csv')

#all_features = M_data.columns.values
#print all_features
X= my_data[1:, 1:-1]
Y = my_data[1:, -1:]

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.33, random_state=42)
feature_name = M_data.columns.values

print train_X.shape
print train_Y.shape
#
## check if any cell is NAN or infinite 
#print np.any(np.isnan(train_X))   #False
#print np.all(np.isfinite(train_X)) #True

#pca = PCA(20)
#pca.fit(train_X)
#x_transform= pca.transform(train_X)
#print x_transform.shape


#
selectf = SelectKBest(f_regression, k = 25)

"""
Support regression 
"""
regressor = LinearSVR(random_state = 30)
para_grid = {"C":[1,2,5,10, 20,50]}
y_pred, tr_score, val_score = pipLine(train_X, train_Y, test_X,  regressor, para_grid, selectf)
plot_error_chart("C", para_grid.get("C"), tr_score, val_score)
#
yy= y_pred.reshape(y_pred.size, 1)

data = np.hstack((test_Y,yy))
print data.shape


df = pd.DataFrame(data)
df.to_csv("prediction_svr25.csv", header = ["Actual", "Prediction"],index = False) 
plt.figure()
plt.plot(test_Y, y_pred, 'r.')
plt.plot([test_Y.min(), test_Y.max()], [test_Y.min(), test_Y.max()], 'b-')
plt.grid(True)
plt.show()
print r2_score(test_Y, y_pred)


"""
Ridge regression 
"""
regressor =Ridge()
para_grid={"alpha": [1, 5, 10, 100, 200]}
y_pred, tr_score, val_score = pipLine(train_X, train_Y, test_X,  regressor, para_grid, selectf)
plot_error_chart("alpha", para_grid.get("alpha"), tr_score, val_score)

yy= y_pred.reshape(y_pred.size, 1)
print test_Y.shape, yy.shape
data = np.hstack((test_Y,yy))
print data.shape


df = pd.DataFrame(data)
df.to_csv("prediction_ridge25.csv", header = ["Actual", "Prediction"],index = False) 
plt.figure(1)
plt.plot(test_Y, y_pred, 'r.')
plt.plot([test_Y.min(), test_Y.max()], [test_Y.min(), test_Y.max()], 'b-')
plt.grid(True)
plt.show()
print r2_score(test_Y, y_pred)


"""
Random Forest
"""
selectf = SelectKBest(f_regression, k = "all")
regressor = RandomForestRegressor(random_state = 30)
para_grid ={"n_estimators": [10,15,20]}
y_pred, tr_score, val_score = pipLine(train_X, train_Y, test_X,  regressor, para_grid, selectf)
plot_error_chart("nestimator", para_grid.get("n_estimators"), tr_score, val_score)

yy= y_pred.reshape(y_pred.size, 1)
print test_Y.shape, yy.shape
data = np.hstack((test_Y,yy))
print data.shape


df = pd.DataFrame(data)
df.to_csv("prediction_randomforest.csv", header = ["Actual", "Prediction"],index = False) 
plt.figure(2)
plt.plot(test_Y, y_pred, 'r.')
plt.plot([test_Y.min(), test_Y.max()], [test_Y.min(), test_Y.max()], 'b-')
plt.grid(True)
plt.show()
print r2_score(test_Y, y_pred)




### Ridge
###regressor = KernelRidge(degree = 2)   #r2 = 0.812
####
####regressor = Lasso()
#
###
###para_grid={"alpha": [1, 5, 10, 100, 200]}
#para_grid ={"n_estimators": [10,15,20]}
#y_pred, tr_score, val_score = pipLine(train_X, train_Y,test_X,  regressor, para_grid, selectf)
#plot_error_chart("alpha", para_grid.get("n_estimators"), tr_score, val_score)
##plt.plot(my_data[1:,0], y_pred, "r-+")
##plt.plot(my_data[1:,0], train_Y, "b-")
#
#pca = PCA(20)
#pca.fit(train_X)
#x_transform= pca.transform(train_X)
##print "shape:  ",  x_transform.shape, train_Y.shape
#regressor = KernelRidge(degree = 2)   
#regressor =Ridge()
#regressor = Lasso()

#regressor = RandomForestRegressor()
#

#para_grid ={"n_estimators": [10,15,20]}

#print test_Y.shape, y_pred.shape




#get_fea_importance(selectf,feature_name, train_X, train_Y)
