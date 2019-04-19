'''
Pipeline for future machine learning projects
'''

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


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

