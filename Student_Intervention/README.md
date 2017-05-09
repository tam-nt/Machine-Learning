## Project Overview

As education has grown to rely more on technology, vast amounts of data has become available for examination and prediction. Logs of student activities, grades, interactions with teachers and fellow students, and more, are now captured in real time through learning management systems like Canvas and Edmodo. This is especially true for online classrooms, which are becoming popular even at the primary and secondary school level. Within all levels of education, there exists a push to help increase the likelihood of student success, without watering down the education or engaging in behaviors that fail to improve the underlying issues. Graduation rates are often the criteria of choice, and educators seek new ways to predict the success and failure of students early enough to stage effective interventions.

A local school district has a goal to reach a 95% graduation rate by the end of the decade by identifying students who need intervention before they drop out of school. As a software engineer contacted by the school district, your task is to model the factors that predict how likely a student is to pass their high school final exam, by constructing an intervention system that leverages supervised learning techniques. The board of supervisors has asked that you find the most effective model that uses the least amount of computation costs to save on the budget. You will need to analyze the dataset on students' performance and develop a model that will predict the likelihood that a given student will pass, quantifying whether an intervention is necessary.

The project is evaluated on three factors:
- F1 score: how well the model can classify whether students who might need early intervention. F1 score is intepreted as a weighted average of the precision and recall: F1 = 2 * (precision * recall) / (precision + recall)
- Size of training data: how much data are required for training classifiers?
- Running time: how long does it take for training and making predictions?

## Software Requirement

This project uses the following software and Python libraries: 

- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn (v0.17)](http://scikit-learn.org/0.17/install.html)

## Code Example

This project contains two files:
- Student_Intervention.py: This is the main code file.
- student-data.csv: The project dataset.

__1. Exploring Data__
- Total number of students: 395
- Number of features: 30
- Number of students who passed: 265
- Number of students who failed: 130
- Graduation rate of the class: 67.09%

__2. Preparing Data__

``` Feature columns:
['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 
'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 
'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 
'Dalc', 'Walc', 'health', 'absences'] 
```

The data contains non-numeric features that need to be converted. Many of them are simply yes/no for example internet. These can be reasonably converted into 1/0 binary values. Some other features have more than 2 values and pandas.get_dummies() function is applied to transform these features.

```
def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))
```

__3. Training and Evaluating Models__

The data is shuffled and split into training and testing sets. Three functions can be used for training and testing the three supervised learning models:
- train_classifier: takes as input a classifier and training data, and fits the classifier to the data.
- predict_labels: takes as input a fit classifier, features, and a target labeling and makes predictions using the F1-score.
- train_predict: takes as input a classifier, and the training and testing dataset, and performs two functions train_classifier() and predict_labels()

```
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
```
In the project, three classifiers are applied:

- Logistic Regression: models the probability that Y response belongs to a particular category.
    + Pros: works well with linearly separable data, requires a small effort of data preparation, is efficient and robust to noise
    + Cons: hardly handles categorical or binary features.    
- Gaussian Naive Bayes
    + Pros: is simple and fast if conditional independence assumption holds, and alleviates well the curse of dimensionality
    + Cons: cant not take into account interactions between features.
- Random forests: an ensemble model provides an improvement over bagged trees. Each time a split in a tree is considered, a random subset of predictors is chosen as split candidates.
    + Pros: performs well with non-linearly separable data and high dimensional problems.
    + Cons: can create a staircase decision boundary and many pieces of hyperplanes.

__4. Selecting the Best Model__

__Logistic Regression__

| Training Set Size  | Training Time | Prediction Time|Training F1 Score | Test F1 Score |
| ------------------ |:-------------:|:--------------:|:----------------:|:-------------:|
| 100                |  0.0015       |  0.0003        |   0.8529	     |  0.7737       |
| 200                |  0.0027	     |  0.0003	      |   0.8269	     |  0.7857       |
| 300                |  0.0033	     |  0.0002	      |   0.8337	     |  0.7368       | 

__Gaussian Naive Bayes__

|Training Set Size	 |Training Time  | Prediction Time| Training F1 Score| Test F1 Score |
| ------------------ |:-------------:|:--------------:|:----------------:|:-------------:|
|  100	             |  0.0007	     |  0.0002	      |  0.8346	         |  0.7402       |
|  200	             |  0.0007	     |  0.0004	      |  0.7879	         |  0.6446       |
|  300	             |  0.0008	     |  0.0004	      |  0.7921	         |  0.6720       |

__Random Forests__

|Training Set Size	 |Training Time  | Prediction Time| Training F1 Score| Test F1 Score |
| ------------------ |:-------------:|:--------------:|:----------------:|:-------------:|
|  100	             |  0.0307	     |  0.0008	      |  0.9767	         |  0.7313       |
|  200	             |  0.027	     |  0.0026	      |  0.9925	         |  0.7519       |
|  300	             |  0.0208	     |  0.0011	      |  0.9924	         |  0.7970       |

Logistic regression and random forests seem to perform well with this data set. F1-scores for testing are greater than 0.7. The random forests produce higher accuracy for traing set of 300 than the logistic regression model, but take pretty much longer time to do training and testing. So it seems appropriate to prefer logistic regression to random forests in term of scalability.

__5. Model Tuning__

Fine tune the chosen model. Use grid search (GridSearchCV) with C parameter for logistic regression model. C is inverse of regularization strength and smaller value indicates stronger regularization.
```
# TODO: Import 'GridSearchCV' and 'make_scorer'
import math
from sklearn import grid_search
from sklearn.metrics import make_scorer, f1_score
# TODO: Create the parameters list you wish to tune
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# TODO: Initialize the classifier
clf = LogisticRegression(random_state=1)

# TODO: Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label = 'yes')

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = grid_search.GridSearchCV(clf, param_grid = parameters, scoring = f1_scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj.fit(X_train, y_train)

# Get the estimator
clf = grid_obj.best_estimator_
```

C of 0.001 provides the best F1 for test data (0.8272) although the training F1 score is smaller (0.7968).
