'''
Pipeline for future machine learning projects
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import seaborn as sns
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn import preprocessing, svm, metrics, tree, decomposition, svm
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import *


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


def impute_values(data):
    '''
    Imputes the median of a given dataframe. If there are
    categorical values the median misses, the mode is imputed
    via value counts
    '''

    data = data.fillna(data.median())

    if sum(data.isnull().sum()):
        data = data.apply(lambda x: x.fillna(x.value_counts().index[0]))

    return data


#Temporal validation function
def split_time(data, start_time, end_time, prediction_windows, time_gap):
    '''
    Splits a time frame into training and testing intervals over a series of
    prediction windows.
    
    Inputs:
        dataframe
        start_time (datetime)
        end_time (datetime)
        prediction_windows (list)
        holdout (int)
    
    Returns
        List of lists of training/testing dates
    '''
    
    #Used as a multiplier to increase training set size over time
    update = 1
    
    #Convert to specified DateTime
    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')

    time_splits = []
    
    #Iteration for an arbitrary number of prediction windows
    for prediction_window in prediction_windows:
        test_end_time = start_time_date
        
        #Subtracting time gap from end_time to ensure properly sized test sets
        while(end_time_date - relativedelta(days=+ time_gap + 1) > test_end_time):

            train_start_time = start_time_date
            
            #Train end time is "start time + 1,2,... * prediction window (e.g. 6 months) - time gap"
            train_end_time = (train_start_time + 
                              update * relativedelta(months=+prediction_window) 
                              - relativedelta(days=+ time_gap + 2))
            
            #Test start time is "train end time + time gap"
            test_start_time = train_end_time + relativedelta(days=+ time_gap + 2)
            
            #Test end time is "test end time - time gap"
            test_end_time = (test_start_time + 
                             relativedelta(months=+prediction_window) 
                             - relativedelta(days=+ time_gap + 2))

            time_splits.append([start_time_date, train_end_time,
                                test_start_time, test_end_time])
            update += 1  
    
    return time_splits


def create_data_splits(data, splits, date):
    '''
    Given a list of temporal splits, data is partitioned into training
    and testing sets

    Inputs:
        data (dataframe)
        splits (list): list of dates
        date (str)

    Returns:
        tuple of dataframes
    '''

    training = data[(data[date] >= splits[0]) &
                    (data[date] <= splits[1])]
 
    testing = data[(data[date] >= splits[2]) &
                   (data[date] <= splits[3])]

    return training, testing


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


def category_to_dummy(data):
    '''
    Converts all categorical variables to dummy variables. Drops the original
    categorical column after conversion
    '''

    for col in data.select_dtypes(include=['category']).columns:
        data = create_binary_vars(data, col)
        data = data.drop([col], axis=1)

    return data


###Build Classifier
def classify(data_splits, classifiers, features, target, parameters, models_to_run, thresholds):
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

    #Structuring results dataframe to be built
    results = pd.DataFrame(columns=('model_type', 'time_period', 'classifier', 'parameters',
                                    'threshold', 'auc-roc', 'precision', 'recall',
                                    'f1_score'))
    
    #Iterating over train/test splits
    for time, data in enumerate(data_splits):
        training, testing = data
        X_train = training[features]
        y_train = training[target]
        
        X_test = testing[features]
        y_test = testing[target]

        #Iterating over given classifiers
        for index, classifier in enumerate([classifiers[x] for x in models_to_run]):
            parameter_values = parameters[models_to_run[index]]

            #Iterates over every combination of parameters for a given classifier
            for p in ParameterGrid(parameter_values):
                try:
                    classifier.set_params(**p)
                    
                    if classifier == 'SVM':
                        y_pred_probs = classifier.fit(X_train, y_train).decision_function(X_test)
                    else:
                        y_pred_probs = classifier.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    
                    #Sorts predicted probabilities and actual outcomes
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    
                    #Iterates over a given list of thresholds and calculates
                    #precision, recall, and the f1 score - then appends a row
                    #to the dataframe
                    for threshold in thresholds:
                        precision = precision_at_k(y_test_sorted, y_pred_probs_sorted, threshold)
                        recall = recall_at_k(y_test_sorted, y_pred_probs_sorted, threshold)
                        f1_score = get_f1_for_k(y_test_sorted, y_pred_probs_sorted, threshold)
                        
                        
                        results.loc[len(results)] = [models_to_run[index], time + 1, classifier, p, threshold,
                                                     roc_auc_score(y_test, y_pred_probs), precision,
                                                     recall, f1_score]
                        
                except IndexError as e:
                    print('Error:', e)
                    continue
    
    return results
                

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


def plot_roc(name, probs, true):
    '''
    Plots the ROC curve for a given dataset.
    Code adapted from Rayid Ghani with permission

    Inputs:
        name (str)
        probs (Y pred probs)
        true (Y test)

    Returns:
        None
    '''
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
    plt.show()

def generate_binary_at_k(y_scores, k):
    '''
    Helper function for precision/recall at k. Converts predicted scores
    to binary outcomes. Code adapted from Rayid Ghani with permission

    Inputs:
        y_scores: array of predicted scores
        k: threshold

    Returns:
        array of binary outcomes
    '''
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]

    return predictions_binary


def precision_at_k(y_true, y_scores, k):
    '''
    Calculates precision for a model at a given k threshold.
    Code adapted from Rayid Ghani with permission.

    Inputs:
        y_true: array of actual scores
        y_scores: array of predicted scores
        k (float): a threshold

    Returns:
        precision (float)
    '''

    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)

    return precision


def recall_at_k(y_true, y_scores, k):
    '''
    Calculates recall for a model at a given k threshold.
    Code adapted from Rayid Ghani with permission.

    Inputs:
        y_true: array of actual scores
        y_scores: array of predicted scores
        k (float): a threshold

    Returns:
        recall (float)
    '''

    preds_at_k = generate_binary_at_k(y_scores, k)
    recall = recall_score(y_true, preds_at_k)

    return recall


def get_f1_for_k(y_true, y_scores, k):
    """
    Calculates the F1 score using the formula F1 = 2 * (precision * recall) / (precision + recall)
    Input:
    y_true: an array of true outcome labels
    y_scores: an array of predicted scores
    k: a threshold proportion
    Returns: An F1 score.
    """

    precision_k = precision_at_k(y_true, y_scores, k)
    recall_k = recall_at_k(y_true, y_scores, k)

    return 2 * (precision_k * recall_k) / (precision_k + recall_k) 



def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    Plots the precision/recall curve for a given dataset.
    Code adapted from Rayid Ghani with permission

    Inputs:
        y_true: array of actual outcome labels
        y_prob: array of predicted probabilities
        model name: str

    Returns:
        None 
    '''
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
    
    plt.title(model_name)
    plt.show()
