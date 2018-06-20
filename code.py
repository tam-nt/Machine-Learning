# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 07:52:22 2017

@author: nttam
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#-----------------------
# loading data
print("Loading data");

try:
    data = pd.read_csv("bank-additional-full.csv", sep = ';')
    print("Dataset contains {} samples with {} variables each".format(*data.shape)) # 4521, 17
except:
    print("Can not load data")
   
# split data into training and testing sets
data = data[data['default'] != 'yes']
y = data['y'] # seaprate target variable
x = data.drop(['y'], axis = 1, inplace = False)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.3, stratify = y, random_state = 1, shuffle = True)

print("Training set has %i samples. Each sample has %i features")%(train_x.shape[0], train_x.shape[1]) # 28831
print("Test set has %i samples. Each sample has %i features")%(test_x.shape[0], test_x.shape[1]) # 12357

train_x = train_x.reset_index(drop = True)
train_y = train_y.reset_index(drop = True)
test_x = test_x.reset_index(drop = True)
test_y = test_y.reset_index(drop = True)

#-----------------------
# visualize data
plt.figure()
plt.figure()
(train_y.value_counts() * 100./train_y.shape[0]).plot(kind = 'bar', fontsize = 14, title = 'Outcome of Campaign Contacts', color = 'orange')
plt.ylabel('Frequency (%)')
plt.tight_layout()

# Separate numeric from categorical
num_data = train_x.select_dtypes(include = ['int64', 'float64'])
cat_data = train_x.select_dtypes(exclude = ['int64', 'float64'])

# Drop 'duration' variable
num_data.drop(['duration'], axis = 1, inplace = True)
num_data.drop(['previous'], axis = 1, inplace = True)
num_data.drop(['pdays'], axis = 1, inplace = True)

# Cacluate skewness of numeric data
print("Skewness of numeric variables")
num_data.skew() # , campaign = 4.86, pdays 1.57, nr.employed -1.04
from matplotlib import pyplot as plt
pd.scatter_matrix(num_data, alpha = 0.3, diagonal = 'kde', figsize = (10, 12))

campaign = num_data['campaign']
campaign = np.log(campaign) # new skew = 0.9
num_data['campaign'] = campaign

# Check correlations
correlations = num_data.corr(method = 'pearson')
print(correlations)
# Plot correlation map
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin = -1, vmax = 1)
fig.colorbar(cax)
ticks = np.arange(0, 14, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
colnames = np.arange(1, 15, 1)
ax.set_xticklabels(colnames)
ax.set_yticklabels(colnames)
plt.show()

# Standardize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
transformed_num_data = pd.DataFrame(scaler.fit_transform(num_data), columns = num_data.keys())

# Convert categorical variables into dummy ones.
dummies_data = pd.get_dummies(cat_data)
column_names = np.concatenate([transformed_num_data.keys(), dummies_data.keys()], axis = 0)
new_train_x = np.column_stack((transformed_num_data, dummies_data))

# Convert target variable into [0, 1]
train_y = train_y.replace('yes', 1)
train_y = train_y.replace('no', 0)

test_y = test_y.replace('yes', 1)
test_y = test_y.replace('no', 0)

#-----------------------
# Preprocess test data for model evaluations.
# preprocess test data: remove duration, previous, pdays, 
#                   log-transform campaign
#                   standardize numeric data
#                   convert categorical var into dummies

test_num_x = test_x.select_dtypes(include = ['int64', 'float64'])
test_cat_x = test_x.select_dtypes(exclude = ['int64', 'float64'])

test_num_x.drop(['duration', 'previous', 'pdays'], inplace = True, axis = 1)
campaign = test_num_x['campaign']
campaign = np.log(campaign)
test_num_x['campaign'] = campaign

scaler = StandardScaler()
transformed_test_num_x = pd.DataFrame(scaler.fit_transform(test_num_x), columns = test_num_x.keys())
test_dummies_x = pd.get_dummies(test_cat_x)
new_test_x = np.column_stack((transformed_test_num_x, test_dummies_x))

#-----------------------
# Start train models with default setting
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier 

from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

models = []
models.append(('DecisionTree', DecisionTreeClassifier(random_state = 1)))
models.append(('AdaBoost', AdaBoostClassifier(random_state = 1)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC(random_state = 1)))
models.append(('NeuralNet', MLPClassifier(random_state = 1)))

from imblearn.combine import SMOTETomek

auc_results = []
training_time = []
names = []
kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
for name, model in models:
    start = time()
    
    auc_scores = []
    print("Training default %s ...")%(name)
    
    for (train_index, val_index) in kfold.split(new_train_x, train_y):
        tr_x, val_x = new_train_x[train_index, :], new_train_x[val_index, :]
        tr_y, val_y = train_y[train_index], train_y[val_index]
        # smote train_x
        sm = SMOTETomek(random_state = 1)
        smote_x, smote_y = sm.fit_sample(tr_x, tr_y)
        model.fit(smote_x, smote_y)
        pred_val_y = model.predict(val_x)
        auc = roc_auc_score(val_y, pred_val_y)        
        auc_scores.append(auc)    
        
    end = time()
    training_time.append(end-start)
    auc_results.append(auc_scores)
    names.append(name)
    print("%s: ")%(name)
    print("CV time: %f")%(end-start)
    print("AUC: %f; Std = %f")%(np.mean(auc_scores), np.std(auc_scores))

# Save default results    
default_auc = pd.DataFrame()
default_auc['Model'] = names
for i in range(1, 11, 1):
    default_auc[str(i)] = (np.transpose(auc_results))[i-1]
    
default_auc.to_csv('default_auc.csv', index = False)
default_results = pd.DataFrame()
default_results['Model'] = names
default_results['Mean_AUC'] = np.mean(auc_results, axis = 1)

default_results.to_csv('mean_default_results.csv', index = False)

# plot auc results    
mean_auc = np.mean(auc_results, axis = 1)
plt.boxplot(auc_results)
plt.title('10-fold Cross Validation AUC')
plt.xticks(range(1, 6), ('Decision Tree', 'AdaBoost', 'KNN', 'SVC', 'Neural Net'))

# plot run times
plt.figure()
x_range = np.arange(len(names))
plt.bar(x_range, training_time, align = 'center')
plt.xticks(x_range, names)
plt.title("Running Time (s)")
plt.ylim(0, 2500)
plt.yticks(np.arange(0, 2510, 500))

#----------------------- 
# Tune decision trees, adaboost and svc with balanced class weight mode first
base_DT = DecisionTreeClassifier(random_state = 2, max_depth = 1, class_weight = 'balanced') # try class_weight with balanced and then can {0: 0.1, 1: 0.9}
models = []
models.append(('DecisionTree', DecisionTreeClassifier(random_state = 1, class_weight = 'balanced'), {'max_depth': [4, 8, 12, 16, 20, 24], 'min_samples_leaf': [0.005, 0.01, 0.02]}))
models.append(('AdaBoost', AdaBoostClassifier(random_state = 1, base_estimator = base_DT), {'n_estimators': [200, 400, 800, 1000, 1200, 1500, 2000, 4000, 6000, 8000], 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1]}))
models.append(('SVC', SVC(random_state = 1, class_weight = 'balanced'), {'C': [0.01, 0.05, 0.1, 0.5, 1, 2], 'gamma': [0.01, 0.05, 0.1, 0.5, 1]}))

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

auc_scorer = make_scorer(roc_auc_score, greater_is_better = True)
kfold = StratifiedKFold(n_splits = 5, random_state = 1, shuffle = True)

results = []
training_time = []
names = []

for name, model, params in models:
    print("Tuning %s ...")%(name)
    start = time()
    grid_obj = GridSearchCV(model, param_grid = params, scoring = auc_scorer, cv = kfold) # statrifiedKfold is used
    grid_obj.fit(new_train_x, train_y)
    grid_obj.best_estimator_
    results.append(grid_obj)
    names.append(name)
    end = time()
    training_time.append(end-start)
    print("Best AUC: %f")%(grid_obj.best_score_)
    print("Best model: %s")%(grid_obj.best_estimator_)
    print("Searching time: %f")%(end-start)

# then tune KNN and Neural Net with SMOTE-Tomek resampled data 
models.append(('KNN', KNeighborsClassifier(), {'n_neighbors': [1, 5, 10, 15, 20, 40, 60, 80, 100, 200, 400, 800, 1000]}))
models.append(('NeuralNet', MLPClassifier(random_state = 1, shuffle = True, verbose = True, max_iter = 2000, learning_rate = 'adaptive'), {'hidden_layer_sizes':[(20,), (40,), (60,), (80,), (100,), (150,)]})) # tuning # of neurons on one-hidden layer network first
models.append(('NeuralNet', MLPClassifier(random_state = 1, shuffle = True, verbose = True, max_iter = 2000, learning_rate = 'adaptive'), {'hidden_layer_sizes':[(150,), (150, 50), (150, 50, 50), (150, 50, 50, 50), (150, 50, 50, 50, 50), (150, 50, 50, 50, 50, 50)]})) # then tuning # of hidden layers
results = []
training_time = []
names = []

for name, model, params in models:
    print("Tuning %s ...")%(name)
    start = time()
    sm = SMOTETomek(random_state = 1)
    smote_x, smote_y = sm.fit_sample(new_train_x, train_y)
    grid_obj = GridSearchCV(model, param_grid = params, scoring = auc_scorer, cv = kfold) # statrifiedKfold is used
    grid_obj.fit(smote_x, smote_y)
    grid_obj.best_estimator_
    results.append(grid_obj)
    names.append(name)
    end = time()
    training_time.append(end-start)
    print("Best AUC: %f")%(grid_obj.best_score_)
    print("Best model: %s")%(grid_obj.best_estimator_)
    print("Searching time: %f")%(end-start)
    
#-----------------------
# Train and evaluate best obtained models 
models = []
models.append(('DecisionTree', DecisionTreeClassifier(random_state = 1, class_weight = 'balanced', max_depth = 8, min_samples_leaf = 0.005)))
models.append(('AdaBoost', AdaBoostClassifier(random_state = 1, base_estimator = base_DT, learning_rate = 1, n_estimators = 4000)))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 1)))
models.append(('SVC', SVC(random_state = 1, C = 0.1, gamma = 0.1, class_weight = 'balanced', verbose = True)))
models.append(('NeuralNet', MLPClassifier(random_state = 1, hidden_layer_sizes = (150, 50, 50, 50 ,50), learning_rate = 'adaptive',max_iter = 2000, verbose = True)))

auc_results = []
training_time = []
names = []
kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
sm = SMOTETomek(random_state = 2)
smote_x, smote_y = sm.fit_sample(new_train_x, train_y)
        
for name, model in models:
    start = time()    
    auc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    confusion_matrix_scores = []
    print("Training default %s ...")%(name)
    if (name in ['KNN', 'NeuralNet']):
        
        for (train_index, val_index) in kfold.split(smote_x, smote_y):
            tr_x, val_x = smote_x[train_index, :], smote_x[val_index, :]
            tr_y, val_y = smote_y[train_index], smote_y[val_index]
        
            model.fit(tr_x, tr_y)
            pred_val_y = model.predict(val_x)
            auc = roc_auc_score(val_y, pred_val_y)
            auc_scores.append(auc)
        end = time()
    else:
        for (train_index, val_index) in kfold.split(new_x_train, train_y):
            tr_x, val_x = new_x_train[train_index, :], new_x_train[val_index, :]
            tr_y, val_y = train_y[train_index], train_y[val_index]        
            model.fit(tr_x, tr_y)
            pred_val_y = model.predict(val_x)
            auc = roc_auc_score(val_y, pred_val_y)
            auc_scores.append(auc)
        end = time()
        
    training_time.append(end-start)
    names.append(name)
    print("******************************")
    print("%s: ")%(name)
    print("CV results: ")
    print("CV time: %f")%(end-start)
    print("AUC: %f; Std = %f")%(np.mean(auc_scores), np.std(auc_scores))
    
    pred_test_y = model.predict(new_test_x)
    test_auc = roc_auc_score(test_y, pred_test_y)
        
    print("-----")
    print("Test results: ")
    print("Test AUC: %f")%(test_auc)
        
    auc_scores.append(test_auc)
    auc_results.append(auc_scores)

# Save to csv file
auc = pd.DataFrame()
auc['Model'] = names
for i in range(1, 12, 1):
    auc[str(i)] = (np.transpose(auc_results))[i-1]
    
auc.to_csv('auc.csv', index = False)

#----------------------------------------
# Plot learning curves for models: use the code for plotting learning curves from http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 20180]):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    #plt.legend(loc=4)
    return plt

title = "Learning Curves (Neural Net)"
cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 1)
#estimator = KNeighborsClassifier(n_neighbors = 1)
estimator = MLPClassifier(random_state = 1, shuffle = True, verbose = True, max_iter = 2000, hidden_layer_sizes = (150, 50, 50, 50), learning_rate = 'adaptive')
sm = SMOTETomek(random_state =1)
smote_x, smote_y = sm.fit_sample(new_train_x, train_y)
plot_learning_curve(estimator, title, smote_x, smote_y, ylim = (0.5, 1.01), cv = cv, n_jobs = 4)
        











