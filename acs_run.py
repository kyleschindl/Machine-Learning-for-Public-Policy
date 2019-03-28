'''
This script downloads data on county-level age, race, and income from the
American Community Survey's 2017 5-year estimates, using the Census API -
specifically, the Python wrapper developed by Datamade
(https://github.com/datamade/census) - and then generates csv
files with ACS demographics data. See acs_base.py for the functions used.
We separate these into different files for a more streamlined experience when
actually performing the download.
'''

import acs_base as base

from census import Census
from us import states
import pandas as pd
from functools import reduce
import random

def df_creation(input_yr):
    c = Census("1006a1e4567f2582e57c3affbab639c19c930271", year=int(input_yr))
    age_df = base.get_ages(c)
    median_age_df = base.get_median_age(c)
    race_df = base.get_races(c)
    hispanic_df = base.get_hispanic(c)
    median_inc_df = base.get_median_inc(c)
    income_df = base.get_incomes(c)
    sexes_df = base.get_sexes(c)
    # We then print out randomly selected counties for some sanity-checking.
    print(age_df.iloc[random.randint(0, 3221)])
    print(median_age_df.iloc[random.randint(0, 3221)])
    print(race_df.iloc[random.randint(0, 3221)])
    print(hispanic_df.iloc[random.randint(0, 3221)])
    print(median_inc_df.iloc[random.randint(0, 3221)])
    print(income_df.iloc[random.randint(0, 3221)])
    print(sexes_df.iloc[random.randint(0, 3221)])
    # We merge these multiple dataframes into a single large dataframe containing all
    # demographic information, and perform the same checking process.
    big_df = reduce(lambda x, y: pd.merge(x, y, on='fips'),
                    [median_age_df, race_df, median_inc_df, sexes_df])
    big_df = pd.merge(big_df, hispanic_df[['fips', 'hispanic']], on='fips',
                      how='inner')
    big_df["hispanic"] = ((big_df["hispanic"].div(big_df["county_total_x"], axis=0))
                          * 100)
    big_df = big_df[['NAME_x', 'fips', 'county_x', 'state_x', 'median_age',
                     'white_alone', 'black_alone', 'native_alone', 'asian_alone',
                     'pacific_alone', 'other_alone', 'two_or_more', 'hispanic',
                     'median_income', 'male', 'female', 'county_total_x']]
    big_df = big_df.loc[:, ~big_df.columns.duplicated()]
    big_df.rename(columns={"NAME_x": "NAME", "county_x": "county", "state_x":
                           "state", "county_total_x": "total_popn"}, inplace=True)
    big_df.iloc[random.randint(0, 3221)]
    suffix = '_' + input_yr
    big_df.columns = [col + suffix if col not in ['NAME', 'fips', 'county', 'state'] 
                      else col for col in big_df.columns]
    print(big_df.columns)
    # Once completed, we write the dataframe to a csv.
    filename = "all_demographics_" + input_yr + ".csv"
    print(filename)
    big_df.to_csv(filename, index=False)
    return big_df

df_2010 = df_creation("2010")
df_2012 = df_creation("2012")
df_2016 = df_creation("2016")

final_df = reduce(lambda x, y: pd.merge(x, y, on='fips'), [df_2010, df_2012, df_2016])
final_df.drop(columns=['NAME_y', 'county_y', 'state_y', 'NAME', 'county', 'state'], 
              inplace=True)
final_df.rename(columns={"NAME_x": "NAME", "county_x": "county", "state_x": "state"}, 
                inplace=True)
final_df.to_csv("all_demographics_2010-2016.csv", index=False)
