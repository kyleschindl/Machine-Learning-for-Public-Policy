'''
Includes code to run and save regression output as txt files
'''

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def clean_csv(filename, columns):
    '''
    Removes states that did not report certain columns in all three years.
    
    Inputs:
        filename (str): a csv file
        columns (list): a list of column names

    Returns:
        df (dataframe)
    '''

    df = pd.read_csv(filename)

    for col in columns:
        df = df[df[col] != 0]

    return df

def create_regression(df, dependent, independent):
    '''
    Creates a summary output for regular OLS regression

    Inputs:
        df (dataframe): a dataframe
        dependent (str): string of column name
        independent (list): a list of column names

    Returns:
        Regression object
    '''

    ind_vars = " + ".join(independent)

    return smf.ols(formula=dependent + " ~ " + ind_vars, data=df).fit()


def write_to_txt(result, filename):
    '''
    Writes the result of a regression output to txt file

    Inputs:
        result (object): result object from regression output
        filename (str): string of desired filename

    Returns:
        None
    '''

    with open(filename, 'w') as f:
        f.write(result.summary().as_text())

data = pd.read_csv("polling_data_for_stats.csv")
region_columns = ["black_alone", "hispanic", "median_income", "republican_2012"]
nation_columns = region_columns + ["clean_region"]

for region in ["Midwest", "South", "Northeast", "West"]:
    result = create_regression(data[data["clean_region"] == region],
                               "polling_loc_pct_change_12_16", region_columns)
    write_to_txt(result, "Regression_Output/" + region + "12-16.txt")


result_2016 = create_regression(data, "polling_loc_pct_change_12_16", nation_columns)
write_to_txt(result_2016, "Regression_Output/nationwide_2016.txt")

result_2008 = create_regression(data, "polling_loc_pct_change_08_12", nation_columns)
write_to_txt(result_2008, "Regression_Output/nationwide_2008.txt")

