'''
Pipeline for future machine learning projects
'''

from __future__ import division
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pylab as pl
from datetime import timedelta
import random
from scipy import optimize
import time
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import itertools


#Updated classifier with parameter experimentation, temporal validation, and
#Additional evaluation metrics
def classify(data, validation_dates, prediction_time, features, target, parameters, models_to_run):
    '''
    Iterates through a sequence of models, prediction windows, and parameters
    in order to build a dataframe of evaluation metrics

    Inputs:
        dataframe
        validation_dates (list of prediction windows)
        prediction_time (date column from dataframe)
        features (list of columns)
        target (target column)
        parameters (dictionary)
        models_to_run (list)

    Returns:
        dataframe
    '''

    results = pd.DataFrame(columns=('model_type', 'clf', 'parameters',
                                    'time_period', 'threshold', 'auc-roc',
                                    'accuracy', 'f1_score','precision', 'recall'))

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
            algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5,
            max_depth=6, n_estimators=10),
        'DT': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'BG': BaggingClassifier()}


    
    for index, clf in enumerate([clfs[x] for x in models_to_run]):
        parameter_values = parameters[models_to_run[index]]
        print(models_to_run[index])
        for p in ParameterGrid(parameter_values):
            for i, validation_date in enumerate(validation_dates):
                print(clf)
                clf.set_params(**p)
                train_set = data[data[prediction_time] <= validation_date[1]]
                validation_set = data[data[prediction_time] >= validation_date[2]]
                
                X_train = train_set[features]
                y_train = train_set[target]
                
                x_test = validation_set[features]
                y_test = validation_set[target]
                
                y_pred_probs = clf.fit(X_train, y_train).predict_proba(x_test)[:,1]

                for threshold in [.01, .02, .05, .1, .2, .3, .5]:
                    pred_label = np.where(y_pred_probs > threshold, 1, 0)
                    precision = precision_score(y_test, pred_label)
                    recall = recall_score(y_test, pred_label)

                    results.loc[len(results)] = [models_to_run[index],clf, p, i, threshold,
                                                 roc_auc_score(y_test, pred_label), 
                                                 accuracy_score(y_test, pred_label),
                                                 f1_score(y_test, pred_label),
                                                 precision, recall]
                                                 
    
    return results


#Temporal validation function
def split_time(data, start_time, end_time, prediction_windows):
    '''
    Splits a time frame into training and testing intervals over a series of
    prediction windows.
    
    Inputs:
        dataframe
        start_time (datetime)
        end_time (datetime)
        prediction_windows (list)
    
    Returns
        List of lists of training/testing dates
    '''

    from datetime import date, datetime, timedelta
    from dateutil.relativedelta import relativedelta
    
    update = 1

    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')

    time_splits = []
    
    for prediction_window in prediction_windows:
        test_end_time = start_time_date
        while(end_time_date > test_end_time):
            train_start_time = start_time_date
            train_end_time = (train_start_time + update * 
                relativedelta(months=+prediction_window) - relativedelta(days=+1))
            test_start_time = train_end_time + relativedelta(days=+1)
            test_end_time = (test_start_time + relativedelta(months=+prediction_window)
                             - relativedelta(days=+1))

            
            time_splits.append([start_time_date, train_end_time, test_start_time, test_end_time])
            update += 1  
    
    return time_splits


'''
The following code is adapted from Rayid Ghani's github with permission:
'''
def plot_roc(name, probs, true, output_type):
    fpr, tpr, thresholds = roc_curve(true, probs)
    roc_auc = auc(fpr, tpr)
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.05])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title(name)
    pl.legend(loc="lower right")
    if (output_type == 'save'):
        plt.savefig(name)
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary


def precision_at_k(y_true, y_scores, k):
    #y_scores_sorted, y_true_sorted = zip(*sorted(zip(y_scores, y_true), reverse=True))
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true_sorted, preds_at_k)
    return precision


def recall_at_k(y_true, y_scores, k):
    #y_scores_sorted, y_true_sorted = zip(*sorted(zip(y_scores, y_true), reverse=True))
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall


def plot_precision_recall_n(y_true, y_prob, model_name, output_type):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    if (output_type == 'save'):
        plt.savefig(name)
    elif (output_type == 'show'):
        plt.show()
    else:
        plt.show()


def get_subsets(l):
    subsets = []
    for i in range(1, len(l) + 1):
        for combo in itertools.combinations(l, i):
            subsets.append(list(combo))
    return subsets


def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]


### Read/Load Data
def clean_csv(csv, index):
    '''
    Reads in a csv file and sets index column.
    
    Inputs:
        csv (str): csv filename
        index (str): column string of index
    Returns:
        dataframe
    '''
    return pd.read_csv(csv, index_col=index)


### Explore Data
def create_heatmap(dataframe, size):
    '''
    Creates and saves a heatmap as a file, given a dataframe, desired size,
    and columns to generate a matrix of correlations
    Inputs:
        dataframe (dataframe): a dataframe
        corr_columns (list): list of column names
        size (tuple): tuple of width, height (ints)
        filename (string): string of desired filename
    Returns:
        None
    '''

    corr_matrix = dataframe.corr().fillna(0)
    x, y = size
    png, ax = plt.subplots(figsize=(x,y))

    ax = sns.heatmap(corr_matrix, center=0, vmin= -1, vmax=1,
        cmap=sns.diverging_palette(250, 10, as_cmap=True))

    plt.show()


def visualize_missing(dataframe, size):
    '''
    Visualize missing data to determine if there are any patterns
    Inputs:
        dataframe
    Outputs:
        heatmap
    '''

    x, y = size
    plt.figure(figsize=(x, y))
    sns.heatmap(dataframe[dataframe.columns[dataframe.isnull().any()
                                                ].tolist()].isnull(), 
            yticklabels=False, cbar=False)

    return plt.show()


def get_nulls(dataframe):
    '''
    Returns series indicating count of missing values
    Inputs:
        dataframe
    Returns:
        series
    '''

    return dataframe.isnull().sum()


def box_and_whisker(dataframe):
    '''
    Returns a boxplot for each column in dataframe in order to visualize any
    outliers present.
    Inputs:
        dataframe
    Returns:
        None
    '''

    for col in dataframe.columns:
        dataframe.boxplot(column=col)
        plt.show()


def get_histograms(dataframe, size):
    '''
    Returns a histogram for each column in dataframe in order to visualize the
    distribution of each variable.
    Inputs:
        dataframe
        size (tuple)
    Returns:
        histogram plot
    '''

    x, y = size
    dataframe.hist(figsize=(x, y))
    plt.tight_layout()
    
    return plt.show()


def get_summaries(dataframe, columns):
    '''
    Returns descriptive statistics of columns
    Inputs:
        dataframe
        columns (list)
    Returns:
        dataframe
    '''

    return dataframe[columns].describe()


###Pre-Process Data
def pre_process(dataframe):
    '''
    Fills missing values with the column median
    Inputs:
        dataframe
    Returns:
        None
    '''
    dataframe.fillna(dataframe.median(), inplace=True)


def convert_tf(dataframe, cols, letter):
    '''
    Coverts columns of strings (such as 't' for True) to values 0, 1

    Inputs:
        dataframe
        cols (list of columns)
        letter: string that represents False

    Returns:
        None
    '''

    for col in cols:
        dataframe[col] = np.where(dataframe[col] == letter, 0, 1)


def convert_to_categorical(dataframe, columns):
    '''
    Converts columns of a dataframe to categorical variables

    Inputs:
        dataframe
        columns (list of column strings)

    Returns:
        None
    '''

    for col in columns:
        dataframe[col] = pd.Categorical(dataframe[col])


###Generate Features/Predictors
def discretize(column, num_bins):
    '''
    Creates various sized bins to discretize a continuous variable
    based on quantile size.
    
    Inputs:
        column (series)
        num_bins (int)
    
    Returns:
        series
    '''
    return pd.qcut(column, num_bins)


def create_binary_vars(dataframe, column):
    '''
    Creates dummy variables from a categorical column
    
    Inputs:
        column (series)
    
    Returns:
        dataframe
    '''
    return pd.concat([dataframe, pd.get_dummies(dataframe[column])], axis=1)


###Build Classifier
def classify_model(features, target):
    '''
    Builds a logistic regression model.
    
    Inputs:
        features (dataframe)
        target (series)
    Returns:
        tuple of series
    '''
    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                 test_size=0.25, random_state=0)
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    return y_test, y_pred
                

###Evaluate Classifier:
def evaluate_classifier(y_test, y_pred):
    '''
    Prints various evaluation metrics
    Inputs:
        y_test
        y_pred
    Returns:
        None
    '''
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("Confusion Matrix: ", metrics.confusion_matrix(y_test, y_pred))

