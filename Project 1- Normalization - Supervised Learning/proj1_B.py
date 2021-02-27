"""
Created on Sat Jun 13 10:52:00 2020

File B - Money Authentication ML
"""

import pandas as pd
from sklearn.metrics import accuracy_score,precision_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



df = pd.read_csv(r"data_banknote_authentication.txt")

train, test = train_test_split(df, test_size=0.2) # Splitting dataset 
train_X = train.iloc[:, 0:4] # training independent variables 
test_X = test.iloc[:, 0:4] # testing independent variable

train_Y = train.iloc[:,4] # training dependent variables 
test_Y = test.iloc[:,4] # testing dependent variable

def classify(classifier,params, classifier_name, train_X, train_Y, test_X, test_Y):
    # This function takes the sklearn classifier data and predicts. Output of function is the accuracy results

    clf = classifier(**params).fit(train_X, train_Y)  # model fitting
    train_pred = clf.predict(train_X) # model predictions of training set
    test_pred = clf.predict(test_X) #  model predictions of testing set
    
    result = pd.DataFrame.from_dict({
        "classifier_name": [classifier_name],
        "train_precision": [precision_score(train_Y, train_pred)],
        "train_accuracy": [accuracy_score(train_Y, train_pred)],
        "test_precision": [precision_score(test_Y, test_pred)],
        "test_accuracy": [accuracy_score(test_Y, test_pred)]
    }).set_index('classifier_name')

    return result

# Configuring all the classification techniques and their parameters
params = {
    "perceptron": [Perceptron, {"random_state": 10, "max_iter" : 5}],
    "logistic_regression": [LogisticRegression, {"random_state": 10, "solver":"lbfgs"}],
    "knn_classifier": [KNeighborsClassifier, {"n_neighbors": 2}],
    "decision_tree": [DecisionTreeClassifier, {"random_state": 100}],
    "random_forest": [RandomForestClassifier, {"random_state": 100, "max_depth": 10}],
    "svm": [SVC, {"random_state": 10, "gamma": "auto"}]
}

# Initializing an empty dataframe
results = pd.DataFrame.from_dict({
    "train_precision": [],
    "train_accuracy": [],
    "test_precision": [],
    "test_accuracy": []
})

# RUN ALL classification techniques
for key, value in params.items():
    results = results.append(
                classify(value[0], value[1], key, train_X, train_Y, test_X, test_Y))
# Displaying the results 
print("results: \n", results.sort_values(["test_accuracy", "train_accuracy"]))

results.to_csv(r'results.csv')