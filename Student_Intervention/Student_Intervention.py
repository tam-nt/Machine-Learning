
# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

#--------------------------------------
# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"

#--------------------------------------
# Data Exploration
# Calculate number of students
n_students = len(student_data)

# Calculate number of features
n_features = len(student_data.columns.difference(["passed"]))

# Calculate passing students
n_passed = len(student_data[student_data['passed'] == 'yes'])

# Calculate failing students
n_failed = len(student_data[student_data["passed"] == "no"])

# Calculate graduation rate
grad_rate = (float(n_passed)/n_students)*100.

# Print the results
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

#--------------------------------------
# Prepare Data
# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1]

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()

#--------------------------------------
# Preprocess Feature Columns
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''

    # Initialize new output DataFrame
    output = pd.DataFrame(index=X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix=col)

            # Collect the revised columns
        output = output.join(col_data)

    return output


X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

#--------------------------------------
# Training and Testing Data Split
from sklearn import cross_validation

# Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# Shuffle and split the dataset into the number of training and testing points above
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all, test_size= float(num_test)/n_students, random_state=1)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

#--------------------------------------
# Set up

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')

def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))

#--------------------------------------
# Implementation: Model Performance Metrics
''' Select three supervised learning models (Logistic Regression, GaussianNB and Random forest classifier) and run the train_predict function for each one
    I train and predict each classifier for three different training set sizes: 100, 200 and 300
    so the number of expected outputs is 9. '''

# Import the three supervised learning models from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import random
from time import time
from sklearn.ensemble import RandomForestClassifier

# Initialize the three models
clf_A = LogisticRegression( random_state = 1)
clf_B = GaussianNB()
clf_C = RandomForestClassifier(random_state=1)

# Set up the training set sizes
for clf in [clf_A, clf_B, clf_C]:
    print "\n{}: \n".format(clf.__class__.__name__)
    for n in [100, 200, 300]:
        train_predict(clf, X_train[:n], y_train[:n], X_test, y_test)

# Choose the best model
# Logistic regression and random forests do well on the data with F1>0.7, however logistic regression runs faster than random forest
#   So the logistic regression is selected for tuning further.
#--------------------------------------
# Model Tuning
# Import 'GridSearchCV' and 'make_scorer'
import math
from sklearn import grid_search
from sklearn.metrics import make_scorer, f1_score
# Create the parameters list you wish to tune
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Initialize the classifier
clf = LogisticRegression(random_state=1)

# Make an f1 scoring function using 'make_scorer'
f1_scorer = make_scorer(f1_score, pos_label = 'yes')

# Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = grid_search.GridSearchCV(clf, param_grid = parameters, scoring = f1_scorer)

# Fit the grid search object to the training data and find the optimal parameters
grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_
print clf

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))


