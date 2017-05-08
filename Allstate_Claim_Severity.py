# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:36:12 2017

@author: Tam Nguyen
"""
"""
CAPSTONE PROJECT: ALL STATE CLAIMS SEVERITY
KAGGLE COMPETITION: https://www.kaggle.com/c/allstate-claims-severity
"""

"""
----------------------------------------------------------------------------------------------------
Step 1: Understand Data
"""
print ("Importing libraries ...")
#import os
import pandas as pd
import numpy as np
from time import time
import xgboost as xgb

# LOADING DATA
print ("Loading data")
#os.chdir("D:/Tam/Study/MachineLearning/Capstone/")
try:
    trainData = pd.read_csv("train.csv")
    print ("Insurance dataset has {} samples with {} feature each. ".format(*trainData.shape))
    testData = pd.read_csv("test.csv")
except:
    print ("Dataset would not loaded!")
#--------------------------------------
# EXPLORE DATA
print('Exploring data')
print(trainData.describe())
print('Type of features: ')
print(trainData.dtypes) # 116 categorical features, 1 integer feature and 15 numeric features
print(trainData.head(5))

# Remove feature 'id' from training and testing sets
xTrain = trainData.drop(['id', 'loss'], axis = 1, inplace = False) # (188318, 130)
yTrain = trainData['loss']
idTest = testData['id']
xTest = testData.drop(['id'], axis = 1, inplace = False) #( 125546, 130)

#----------------------
# PLOT PEARSON CORRELATION MATRIX MAP: >0.8 IS SIGNIFICANT 
from matplotlib import pyplot
correlations = xTrain.corr(method = 'pearson')
print(correlations)

fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin = -1, vmax = 1)
fig.colorbar(cax)
ticks = np.arange(0, 14, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
colnames = np.arange(1, 15, 1)
ax.set_xticklabels(colnames)
ax.set_yticklabels(colnames)
pyplot.show()
# COMMENT: high correlations btw cont1-cont9, cont1-cont10, cont6-cont10, cont6-cont13, cont11-cont12

"""
----------------------------------------------------------------------------------------------------
STEP 2: PROCESS DATA
PROCESS NUMERIC FEATURES:
    1. Identify the high correlated features: between cont1-cont9, cont1-cont10, cont6-cont10, cont6-cont13, cont11-cont12
    2. Then remove features cont9, cont10, cont11 and cont13 from data set
    3. Examine skewness of the features and transform them if needed: Sknewness is small for continous features except the response "loss"
        Skewness of "loss" is ...., therefore the response needs to be transformed, using logarithm function
PROCESS CATEGORICAL FEATURES:
    1. Remove unbalanced two-value categorical features: the chance of one value is higher than 99%
    2. For categorical feature with the high number of levels, group the levels with the occurence chance between 0 and 1% into a new level of 'rare'
    3. Check inconsistency in levels from categorical features in the training and testing sets.
    4. Transform categorical features into dummy features using get_dummies() function
"""
# PROCESS NUMERIC FEATURES
# Remove high correlated features: cont9, cont10, cont11 and cont13 from the training set and testing set
xTrain.drop(['cont9', 'cont10', 'cont11', 'cont13'], axis = 1, inplace = True) #(188318, 126)
xTest.drop(['cont9', 'cont10', 'cont11', 'cont13'], axis = 1, inplace = True) #(125546, 126)

# Plot histogram of numeric features
xTrain[['cont1', 'cont2', 'cont3', 'cont4']].hist()
xTrain[['cont5', 'cont6', 'cont7', 'cont8']].hist()
xTrain[['cont12', 'cont14']].hist()
pyplot.show()

# Plot scatter matrix
from pandas.tools.plotting import scatter_matrix
scatter_matrix(X_train)
pyplot.show()

# Measure skew of data
skew = xTrain.skew()
print(skew)
print(yTrain.skew())

xTrain.plot(kind = 'box', subplots = True, sharex = False, sharey = True, layout = (2,5))

# Transform the training responsen Y
newYTrain = np.log(yTrain + 200)

#--------------------------------
# PROCESS CATEGORICAL FEATURES
catTrain = xTrain.select_dtypes(exclude=['float64'])
catTest = xTest.select_dtypes(exclude=['float64'])
columnNames = catTrain.columns.values

# Remove unbalanced categorical variables with criterion of 99% for one of two levels 
columnRemove = []
for cat in columnNames:
    if True in list(catTrain[cat].value_counts()/1883.18 > 99.0):
        columnRemove.append(cat) 

catTrain.drop(columnRemove, axis = 1, inplace = True)
catTest.drop(columnRemove, axis = 1, inplace = True)

# Group levels which has frequency between 0 and 1 and call 'rare' level
updatedColumnNames = catTrain.columns.values

# for categorical features from the training set        
for column in updatedColumnNames:
    print (column)
    combinedLevel = []
    percentages = catTrain[column].value_counts() * 100/catTrain.shape[0]
    levels = catTrain[column].unique()
    combinedLevel = [level for level in levels if percentages[level] <= 1.0]
    if len(combinedLevel)>0:
        columnSeries = catTrain[column].replace(combinedLevel, 'rare', inplace = False)
        catTrain[column] = columnSeries

# for categorical features from the testing set
for column in updatedColumnNames:
    print (column)
    combinedLevel = []
    percentages = catTest[column].value_counts() * 100/catTest.shape[0]
    levels = catTest[column].unique()
    combinedLevel = [level for level in levels if percentages[level] <= 1.0]
    if len(combinedLevel)>0:
        catTest[column].replace(combinedLevel, 'rare', inplace = True)

# Combine categorical features from the training and testing sets to eliminate inconsitent levels 
combinedCat = pd.concat([catTrain, catTest])

for column in catTrain.columns.values:
    if catTrain[column].nunique() != catTest[column].nunique():
        print (column)
        trainLevels = catTrain[column].unique()
        testLevels = catTest[column].unique()
        
        trainLevelRemove = set(trainLevels) - set(testLevels)
        testLevelRemove = set(testLevels) - set(trainLevels)
        print (trainLevels)
        print (testLevels)
        print (trainLevelRemove)
        print (testLevelRemove)
        levelRemove = (set(trainLevels) - set(testLevels)).union(set(testLevels) - set(trainLevels))
        def remove_level(x):
                if x in levelRemove:
                    return np.nan
                return x
        combinedCat[column] = combinedCat[column].apply(lambda x: remove_level(x), 1)
        print (catTrain[column].value_counts())
        print (catTest[column].value_counts())
        
#---------------------------------
# TRANSFORM CATEGORICAL FEATURES USING GET_DUMMIES 
transformedCat = pd.get_dummies(combinedCat)

# Get new training and testing sets
newXTrain = np.concatenate((transformedCat.iloc[0: len(xTrain), :], xTrain.select_dtypes(include=['float64'])), axis = 1 )
newXTest = np.concatenate((transformedCat.iloc[len(xTrain) : len(transformedCat), :], xTest.select_dtypes(include = ['float64'])), axis = 1)

"""
----------------------------------------------------------------------------------------------------
STEP 3: MODEL SELECTION AND TRAINING
1. Apply linear and non-linear single models to get the baseline errors
2. Apply ensemble models: random forests and extreme gradient boosting regressor (XGBoost) and compare results from two ensemble models
3. Tune the better ensemble model: XGBoost model
"""
# SINGLE MODELS: LINEAR REGRESSION, RIDGE REGRESSION, LASSO REGRESSION
#                KNN, DECISION TREE AND SUPPORT VECTOR MACHINE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from time import time
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold

models = []
models.append(('LinearRegression', LinearRegression()))
models.append(('Lasso', Lasso()))
models.append(('Ridge', Ridge()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('DescisionTree', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

# 10-FOLD CROSS-VALIDATION
results = []
names = []
training_time = []
for name, model in models:
    start = time()
    kfold = KFold(newXTrain.shape[0], 10)
    scores = []
   
    for i, (trainIndex, testIndex) in enumerate(kfold):
        x_train, x_test = newXTrain[trainIndex, :], newXTrain[testIndex, :]
        y_train, y_test = newYTrain[trainIndex], newYTrain[testIndex]
        model.fit(x_train, y_train)
        predictedY = model.predict(x_test)
        print (predictedY)
        score =  mean_absolute_error(np.exp(predictedY)-200, np.exp(y_test)-200)
        scores.append(score)    
    end = time()
    print ("%s: %f \n Training time: %f")%(name, np.mean(score), end - start)
    names.append(name)
    results.append(scores)
    training_time.append(end-start)    
    
# save results to csv file    
maeResults = pd.DataFrame()
maeResults['algorithm'] = names
maeResults['mean_mae'] = np.mean(results, axis = 1)
maeResults['std_mae'] = np.std(results, axis = 1)
maeResults['training_time (s)'] = training_time
maeResults.to_csv('meanMAE_singlemodels.csv', index = False)    # save averaged MAE, std of MAE, training time from algorithms

pd.DataFrame(results).to_csv('10fold_mae_singlemodels.csv', index = False) # save 10-fold cross-validation results

# plot comparison of training time and mean abosulte errors between models
# plot mean absolute error 
from matplotlib import pyplot
fig = pyplot.figure()
fig.suptitle('Cross-validation Errors')
ax = fig.add_subplot(111)
pyplot.boxplot(np.transpose(results))
ax.set_xticklabels(names)
ax.set_ylim([1100, 1900])
pyplot.show()

# plot training time 
fig = pyplot.figure()
fig.suptitle('Training Time (seconds)')
ax = fig.add_subplot(111)
pyplot.bar(range(0, 6), training_time, color = 'b', width = 1/1.5, align = 'center')
ax.set_xticks(range(0, 6))
ax.set_xticklabels(names)
pyplot.show()

#--------------------------------------------------------------
# APPLY RANDOM FORESTS AND XGBOOST MODEL, THEN COMPARE RESULTS
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error

ensemble_models = []
ensemble_models.append(('Randomforest', RandomForestRegressor(verbose = 1, random_state = 12)))
ensemble_models.append(('XGBoostRegressor', XGBRegressor()))

results = []
names = []
train_time = []

for name, model in ensemble_models:
    start = time()
    kfold = KFold(newXTrain.shape[0], 10)
    scores = []
    for i, (trainIndex, testIndex) in enumerate(kfold):
        x_train, x_test = newXTrain[trainIndex, :], newXTrain[testIndex, :]
        y_train, y_test = newYTrain[trainIndex], newYTrain[testIndex]
        model.fit(x_train, y_train)
        predictedY = model.predict(x_test)
        print (predictedY)
        score =  mean_absolute_error(np.exp(predictedY)-200, np.exp(y_test)-200)
        scores.append(score)
    end = time()
    print ("%s: %f \n Training time: %f")%(name, np.mean(score), end - start)
    names.append(name)
    results.append(scores)
    train_time.append(end-start)

maeResults = pd.DataFrame()
maeResults['algorithm'] = names
maeResults['mean_mae'] = np.mean(results, axis = 1)
maeResults['std_mae'] = np.std(results, axis = 1)
maeResults['training_time (s)'] = train_time
maeResults.to_csv('RF_xgb.csv', index = False) # save averaged MAE, std of MAE, and training time from random forest regressor  and xgboost model

model_results = pd.DataFrame()
model_results = pd.DataFrame(results)
model_results.to_csv('10fold_RFnfeatures_xgb.csv', index = False) # save 10-fold cross-validation results

#--------------------------------------------------------------
# APPLY EXTREME GRADIENT BOOSTING REGRESSORS XGBoost
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.grid_search import GridSearchCV

# Use customized loss and objective function
def mae(y_pred, newYTrain):
    return mean_absolute_error(np.exp(y_pred)-200, np.exp(newYTrain)-200)

fair_constant = 0.7
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess

def mae_function(y_pred, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(y_pred)-200, np.exp(labels)-200)    

#------------------------------------------------
# Estimate the best iteration using xgboost.cv
start = time()
trainDmat = xgb.DMatrix(newXTrain, label = newYTrain)
testDmat = xgb.DMatrix(newXTest)
params = {'learning_rate': 0.1, 'n_estimators': 1000, 'objective' : 'reg:linear',
                  'max_depth': 4, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 1,
                  'seed': 10}
xgbModel = XGBRegressor(params)
cv = xgb.cv(params, dtrain = trainDmat, nfold = 5, num_boost_round = 10000, feval = mae, obj = fair_obj,
            early_stopping_rounds = 100, verbose_eval = True)
end = time()
print ("Cross validation time: ", end - start)
# best iteration is 1542

# Tune max_depth and min_child_weight using GridSearchCV 
cv_params = {'max_depth': [4, 6, 8, 10], 'min_child_weight': [1, 3, 5, 7]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1542, 'seed': 12, 'subsample': 0.8, 'objective': 'reg:linear',
              'max_depth': 4, 'min_child_weight': 3, 'silent': False }

start = time()
optimal_GBM = GridSearchCV(XGBRegressor(**ind_params),
                            cv_params,
                             scoring = make_scorer(mae, greater_is_better = False), cv = 5)

optimal_GBM.fit(newXTrain, newYTrain)
end = time()
print ("Training model grid search in {} seconds".format(end - start))
print (optimal_GBM.best_estimator_) # estimator chosen by the search
print (optimal_GBM.best_score_) # score of best estimator on the left out data
print (optimal_GBM.grid_scores_)

# Tune learning rate and subsample
cv_params = {'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.7, 0.8, 0.9, 1]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1542, 'seed': 12, 'subsample': 0.8, 'objective': 'reg:linear',
              'max_depth': 4, 'min_child_weight': 1, 'silent': False, 'gamma': 0.4}
start = time()
optimal_GBM = GridSearchCV(XGBRegressor(**ind_params),
                            cv_params,
                             scoring = make_scorer(mae, greater_is_better = False), cv = 5)
optimal_GBM.fit(newXTrain, newYTrain)
end = time()
print ("Training model grid search in {} seconds".format(end - start))
print (optimal_GBM.best_estimator_) # estimator chosen by the search
print (optimal_GBM.best_score_) # score of best estimator on the left out data
print (optimal_GBM.grid_scores_)

# Tune gamma:
cv_params = {'gamma': [0, 0.1, 0.2, 0.3, 0.4]}
ind_params = {'learning_rate': 0.05, 'n_estimators': 1542, 'seed': 12, 'subsample': 0.7, 'objective': 'reg:linear',
              'max_depth': 4, 'min_child_weight': 1, 'silent': False, 'gamma': 0 }
start = time()
optimal_GBM = GridSearchCV(XGBRegressor(**ind_params),
                            cv_params,
                             scoring = make_scorer(mae, greater_is_better = False), cv = 5)
optimal_GBM.fit(newXTrain, newYTrain)
end = time()
print ("Training model grid search in {} seconds".format(end - start))
print (optimal_GBM.best_estimator_) # estimator chosen by the search
print (optimal_GBM.best_score_) # score of best estimator on the left out data
print (optimal_GBM.grid_scores_)

# check best iteration again
start = time()
trainDmat = xgb.DMatrix(newXTrain, label = newYTrain)
testDmat = xgb.DMatrix(newXTest)
params = {'learning_rate': 0.05, 'objective' : 'reg:linear', 'gamma': 0.4,
                  'max_depth': 4, 'min_child_weight': 1, 'subsample': 0.7, 'colsample_bytree': 1,
                  'seed': 10}
xgbModel = XGBRegressor(params)
cv = xgb.cv(params, dtrain = trainDmat, nfold = 5, num_boost_round = 10000, feval = mae_function, obj = fair_obj,
            early_stopping_rounds = 100, verbose_eval = True)
end = time()
print ("Cross validation time: ", end - start)
# best iteration is 2643

# Train and make prediction from xgboost model with 10-fold cross-validation
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error

kfold = KFold(newXTrain.shape[0], 10)
testDmat = xgb.DMatrix(newXTest)
totalMaeValue = 0
totalPredictedY = 0
numRounds = []

start = time()
for i, (trainIndex, testIndex) in enumerate(kfold):
    x_train, x_validate = newXTrain[trainIndex, :], newXTrain[testIndex, :]
    y_train, y_validate = newYTrain[trainIndex], newYTrain[testIndex]
    trainDmat = xgb.DMatrix(x_train, label = y_train)
    validateDmat = xgb.DMatrix(x_validate, label = y_validate)
    params = {'learning_rate': 0.05, 'objective' : 'reg:linear',
                  'max_depth': 4, 'min_child_weight': 1, 'subsample': 0.7, 'colsample_bytree': 1,
                  'seed': 10, 'gamma': 0.4}
    xgbModel = xgb.train(params, trainDmat, num_boost_round= 2643, early_stopping_rounds=100,
                          obj=fair_obj, feval=mae_function, verbose_eval=True, evals=[(trainDmat, 'train'), (validateDmat, 'eval')])
    predictedValidateY = xgbModel.predict(validateDmat, ntree_limit = xgbModel.best_ntree_limit)
    maeValue = mean_absolute_error(np.exp(y_validate), np.exp(predictedValidateY))
    print 'Validation MAE: {}'.format(maeValue)
    predictedYTest = np.exp(xgbModel.predict(testDmat, ntree_limit=xgbModel.best_ntree_limit))-200
    # calculate sum of val-mae and sum_y_pred for averaging results later
    totalMaeValue += maeValue
    totalPredictedY += predictedYTest
    numRounds.append(xgbModel.best_iteration)

meanPredictedYTest = totalPredictedY / 10
print 'Mean validation MAE: {}'.format(totalMaeValue/10)
print 'Mean number boost round: {}'.format(np.mean(numRounds))
end = time()
print "Training model grid search in {} seconds".format(end - start)

# Save results to csv file
results = pd.DataFrame()
results['id'] = idTest
results['loss'] = meanPredictedYTest
results.to_csv('xgb_prediction.csv', index = False)
