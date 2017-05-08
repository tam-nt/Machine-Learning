# -*- coding: utf-8 -*-
"""
CAPSTONE PROJECT: ALL STATE CLAIMS SEVERITY
KAGGLE COMPETITION: https://www.kaggle.com/c/allstate-claims-severity
"""

"""
----------------------------------------------------------------------------------------------------
Step 1: Understand Data
"""
print ("Importing libraries ...")
import pandas as pd
import numpy as np
from time import time

# LOADING DATA
print ("Loading data")
try:
    train_data = pd.read_csv("train.csv")
    print ("Insurance dataset has {} samples with {} feature each. ".format(*train_data.shape))
    test_data = pd.read_csv("test.csv")
except:
    print ("Dataset would not loaded!")
#--------------------------------------
# EXPLORE DATA
print('Exploring data')
print(train_data.describe())
print('Type of features: ')
print(train_data.dtypes) # 116 categorical features, 1 integer feature and 15 numeric features
print(train_data.head(5))

# Remove feature 'id' from training and testing sets
x_train = train_data.drop(['id', 'loss'], axis = 1, inplace = False) # (188318, 130)
y_train = train_data['loss']
id_test = test_data['id']
x_test = test_data.drop(['id'], axis = 1, inplace = False) #( 125546, 130)

#----------------------
# PLOT PEARSON CORRELATION MATRIX MAP: >0.8 IS SIGNIFICANT 
from matplotlib import pyplot
correlations = x_train.corr(method = 'pearson')
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
x_train.drop(['cont9', 'cont10', 'cont11', 'cont13'], axis = 1, inplace = True) #(188318, 126)
x_test.drop(['cont9', 'cont10', 'cont11', 'cont13'], axis = 1, inplace = True) #(125546, 126)

# Plot histogram of numeric features
x_train[['cont1', 'cont2', 'cont3', 'cont4']].hist()
x_train[['cont5', 'cont6', 'cont7', 'cont8']].hist()
x_train[['cont12', 'cont14']].hist()
pyplot.show()

# Plot scatter matrix
from pandas.tools.plotting import scatter_matrix
scatter_matrix(x_train)
pyplot.show()

# Measure skew of data
skew = x_train.skew()
print(skew)
print(y_train.skew())

x_train.plot(kind = 'box', subplots = True, sharex = False, sharey = True, layout = (2,5))

# Transform the training responsen Y
new_y_train = np.log(y_train + 200)

#--------------------------------
# PROCESS CATEGORICAL FEATURES
cat_train = x_train.select_dtypes(exclude=['float64'])
cat_test = x_test.select_dtypes(exclude=['float64'])
column_names = cat_train.columns.values

# Remove unbalanced categorical variables with criterion of 99% for one of two levels 
column_remove = []
for cat in column_names:
    if True in list(cat_train[cat].value_counts()/1883.18 > 99.0):
        column_remove.append(cat)

cat_train.drop(column_remove, axis = 1, inplace = True)
cat_test.drop(column_remove, axis = 1, inplace = True)

# Group levels which has frequency between 0 and 1 and call 'rare' level
updated_column_names = cat_train.columns.values

# for categorical features from the training set        
for column in updated_column_names:
    print (column)
    combined_level = []
    percentages = cat_train[column].value_counts() * 100/cat_train.shape[0]
    levels = cat_train[column].unique()
    combined_level = [level for level in levels if percentages[level] <= 1.0]
    if len(combined_level)>0:
        column_series = cat_train[column].replace(combined_level, 'rare', inplace = False)
        cat_train[column] = column_series

# for categorical features from the testing set
for column in updated_column_names:
    print (column)
    combined_level = []
    percentages = cat_test[column].value_counts() * 100/cat_test.shape[0]
    levels = cat_test[column].unique()
    combined_level = [level for level in levels if percentages[level] <= 1.0]
    if len(combined_level)>0:
        cat_test[column].replace(combined_level, 'rare', inplace = True)

# Combine categorical features from the training and testing sets to eliminate inconsitent levels 
combined_cat = pd.concat([cat_train, cat_test])

for column in cat_train.columns.values:
    if cat_train[column].nunique() != cat_test[column].nunique():
        print (column)
        train_levels = cat_train[column].unique()
        test_levels = cat_test[column].unique()
        
        train_level_remove = set(train_levels) - set(test_levels)
        test_level_remove = set(test_levels) - set(train_levels)
        level_remove = (set(train_levels) - set(test_levels)).union(set(test_levels) - set(train_levels))
        def remove_level(x):
                if x in level_remove:
                    return np.nan
                return x
        combined_cat[column] = combined_cat[column].apply(lambda x: remove_level(x), 1)
        print (cat_train[column].value_counts())
        print (cat_test[column].value_counts())
        
#---------------------------------
# TRANSFORM CATEGORICAL FEATURES USING GET_DUMMIES 
transformed_cat = pd.get_dummies(combined_cat)

# Get new training and testing sets
new_x_train = np.concatenate((transformed_cat.iloc[0: len(x_train), :], x_train.select_dtypes(include=['float64'])), axis = 1 )
new_x_test = np.concatenate((transformed_cat.iloc[len(x_train) : len(transformed_cat), :], x_test.select_dtypes(include = ['float64'])), axis = 1)

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
    kfold = KFold(new_x_train.shape[0], 10)
    scores = []
   
    for i, (train_index, validate_index) in enumerate(kfold):
        x_train, x_validate = new_x_train[train_index, :], new_x_train[validate_index, :]
        y_train, y_validate = new_y_train[train_index], new_y_train[validate_index]
        model.fit(x_train, y_train)
        predicted_y_validate = model.predict(x_validate)
        print (predicted_y_validate)
        score =  mean_absolute_error(np.exp(predicted_y_validate)-200, np.exp(y_validate)-200)
        scores.append(score)    
    end = time()
    print ("%s: %f \n Training time: %f")%(name, np.mean(score), end - start)
    names.append(name)
    results.append(scores)
    training_time.append(end-start)    

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
    kfold = KFold(new_x_train.shape[0], 10)
    scores = []
    for i, (train_index, validate_index) in enumerate(kfold):
        x_train, x_validate = new_x_train[train_index, :], new_x_train[validate_index, :]
        y_train, y_validate = new_y_train[train_index], new_y_train[valide_index]
        model.fit(x_train, y_train)
        predicted_y_validate = model.predict(x_validate)
        score =  mean_absolute_error(np.exp(predicted_y_validate)-200, np.exp(y_validate)-200)
        scores.append(score)
    end = time()
    print ("%s: %f \n Training time: %f")%(name, np.mean(score), end - start)
    names.append(name)
    results.append(scores)
    train_time.append(end-start)

mae_results = pd.DataFrame()
mae_results['algorithm'] = names
mae_results['mean_mae'] = np.mean(results, axis = 1)
mae_results['std_mae'] = np.std(results, axis = 1)
mae_results['training_time (s)'] = train_time
#--------------------------------------------------------------
# APPLY EXTREME GRADIENT BOOSTING REGRESSORS XGBoost
from sklearn.grid_search import GridSearchCV

# Use customized loss and objective function
def mae(y_pred, new_y_train):
    return mean_absolute_error(np.exp(y_pred)-200, np.exp(new_y_train)-200)

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
train_dmat = xgb.DMatrix(new_x_train, label = new_y_train)
test_dmat = xgb.DMatrix(new_x_test)
params = {'learning_rate': 0.1, 'n_estimators': 1000, 'objective' : 'reg:linear',
                  'max_depth': 4, 'min_child_weight': 1, 'subsample': 0.8, 'colsample_bytree': 1,
                  'seed': 10}
cv = xgb.cv(params, dtrain = train_dmat, nfold = 5, num_boost_round = 10000, feval = mae, obj = fair_obj,
            early_stopping_rounds = 100, verbose_eval = True)
end = time()
print ("Cross validation time: ", end - start)
# best iteration is 1542

# Tune max_depth and min_child_weight using GridSearchCV 
cv_params = {'max_depth': [4, 6, 8, 10], 'min_child_weight': [1, 3, 5, 7]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1542, 'seed': 12, 'subsample': 0.8, 'objective': 'reg:linear',
              'max_depth': 4, 'min_child_weight': 3, 'silent': False }

start = time()
xgb_model = GridSearchCV(XGBRegressor(**ind_params),
                            cv_params,
                             scoring = make_scorer(mae, greater_is_better = False), cv = 5)

xgb_model.fit(new_x_train, new_y_train)
end = time()
print ("Training model grid search in {} seconds".format(end - start))
print (xgb_model.best_estimator_) # estimator chosen by the search
print (xgb_model.best_score_) # score of best estimator on the left out data
print (xgb_model.grid_scores_)

# Tune learning rate and subsample
cv_params = {'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.7, 0.8, 0.9, 1]}
ind_params = {'learning_rate': 0.1, 'n_estimators': 1542, 'seed': 12, 'subsample': 0.8, 'objective': 'reg:linear',
              'max_depth': 4, 'min_child_weight': 1, 'silent': False, 'gamma': 0.4}
start = time()
xgb_model = GridSearchCV(XGBRegressor(**ind_params),
                            cv_params,
                             scoring = make_scorer(mae, greater_is_better = False), cv = 5)
xgb_model.fit(new_x_train, new_y_train)
end = time()
print ("Training model grid search in {} seconds".format(end - start))
print (xgb_model.best_estimator_) # estimator chosen by the search
print (xgb_model.best_score_) # score of best estimator on the left out data
print (xgb_model.grid_scores_)

# Tune gamma:
cv_params = {'gamma': [0, 0.1, 0.2, 0.3, 0.4]}
ind_params = {'learning_rate': 0.05, 'n_estimators': 1542, 'seed': 12, 'subsample': 0.7, 'objective': 'reg:linear',
              'max_depth': 4, 'min_child_weight': 1, 'silent': False, 'gamma': 0 }
start = time()
xgb_model = GridSearchCV(XGBRegressor(**ind_params),
                            cv_params,
                             scoring = make_scorer(mae, greater_is_better = False), cv = 5)
xgb_model.fit(new_x_train, new_y_train)
end = time()
print ("Training model grid search in {} seconds".format(end - start))
print (xgb_model.best_estimator_) # estimator chosen by the search
print (xgb_model.best_score_) # score of best estimator on the left out data
print (xgb_model.grid_scores_)

# check best iteration again
start = time()
params = {'learning_rate': 0.05, 'objective' : 'reg:linear', 'gamma': 0.4,
                  'max_depth': 4, 'min_child_weight': 1, 'subsample': 0.7, 'colsample_bytree': 1,
                  'seed': 10}
cv = xgb.cv(params, dtrain = train_dmat, nfold = 5, num_boost_round = 10000, feval = mae_function, obj = fair_obj,
            early_stopping_rounds = 100, verbose_eval = True)
end = time()
print ("Cross validation time: ", end - start)

# Train and make prediction from xgboost model with 10-fold cross-validation
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error

kfold = KFold(newXTrain.shape[0], 10)
total_mae = 0
total_predicted_y = 0
num_rounds = []

start = time()
for i, (train_index, validate_index) in enumerate(kfold):
    x_train, x_validate = new_x_train[train_index, :], new_x_train[validate_index, :]
    y_train, y_validate = new_y_train[train_index], new_y_train[validate_index]
    train_dmat = xgb.DMatrix(x_train, label = y_train)
    validate_dmat = xgb.DMatrix(x_validate, label = y_validate)
    params = {'learning_rate': 0.05, 'objective' : 'reg:linear',
                  'max_depth': 4, 'min_child_weight': 1, 'subsample': 0.7, 'colsample_bytree': 1,
                  'seed': 10, 'gamma': 0.4}
    xgb_model = xgb.train(params, train_dmat, num_boost_round= 2643, early_stopping_rounds=100,
                          obj=fair_obj, feval=mae_function, verbose_eval=True, evals=[(train_dmat, 'train'), (validate_dmat, 'eval')])
    predicted_y_validate = xgb_model.predict(validate_dmat, ntree_limit = xgb_model.best_ntree_limit)
    mae_value = mean_absolute_error(np.exp(y_validate), np.exp(predicted_y_validate))
    print 'Validation MAE: {}'.format(mae_value)
    predicted_y_test = np.exp(xgb_model.predict(test_dmat, ntree_limit=xgb_model.best_ntree_limit))-200
    # calculate sum of val-mae and sum_y_pred for averaging results later
    total_mae += mae_value
    total_predicted_y += predicted_y_test
    num_rounds.append(xgb_model.best_iteration)

mean_predicted_y_test = total_predicted_y / 10
print 'Mean validation MAE: {}'.format(total_mae/10)
print 'Mean number boost round: {}'.format(np.mean(num_rounds))
end = time()
print "Training model grid search in {} seconds".format(end - start)


