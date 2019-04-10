'''
Functions for downloading data on county-level age, race, and income
from the American Community Survey's 2017 5-year estimates,
using the Census API - specifically, the Python wrapper developed by Datamade:
https://github.com/datamade/census
'''

from functools import reduce
import pandas as pd


def get_inputs(table_name, lower, upper):
    '''
    Getting a range of consecutively-numbered columns to be pulled from a
    American Community Survey table, which can then be passed to an API call.

    Inputs:
    table_name (str): name of a table in the American Community Survey data
    lower (int): the unique number identifying the first column in the desired
    range
    upper (int): the unique number identifying the last column in the desired
    range

    Returns:
    input_tup (tuple): a tuple of columns to be pulled from an American
    Community Survey table
    '''
    input_lst = []
    for i in range(lower, upper):
        if i < 10:
            istr = table_name + '_00' + str(i) + 'E'
        else:
            istr = table_name + '_0' + str(i) + 'E'
        input_lst.append(istr)
    full_lst = ['NAME'] + input_lst
    input_tup = tuple(full_lst)
    return input_tup


def get_percents(df, count_col_prefix, label_dict):
    '''
    Converting a dataframe of counts to percentages of the total county
    population.

    Inputs:
    df: a dataframe of counts
    count_col_prefix (str): the name prefix identifying the columns that are to
    be converted to percentages
    label_dict (dictionary): a dictionary to rename the dataframe columns

    Returns:
    kpct: a dataframe of percentages
    '''
    df["state"] = df["state"].astype("int")
    df["state"] = df["state"].astype("str")
    df["fips"] = df["state"] + df["county"]
    filter_col = [col for col in df.columns if col.startswith(count_col_prefix)]
    filter_lst = ['NAME', 'fips', 'county', 'state'] + filter_col
    kp = df[filter_lst]
    kp.rename(columns=label_dict, inplace=True)
    to_sum = list(kp)
    to_sum.remove("NAME")
    to_sum.remove("fips")
    to_sum.remove("county")
    to_sum.remove("state")
    kp["county_total"] = kp[to_sum].sum(axis=1)
    kpct = kp.copy()
    kpct.iloc[:, 4:-1] = ((kpct.iloc[:, 4:-1].div(kp["county_total"], axis=0))
                          * 100)
    return kpct


def get_ages(conn):
    '''
    Gets a dataframe showing the percentage of county population in each age
    category.

    Inputs:
    conn: a Census object which connects to the census API

    Returns:
    fin: a dataframe of county-level age distribution
    '''
    agegrp_lst = []
    make_agegrp_input(agegrp_lst)
    dflst = []
    for agegrp in agegrp_lst:
        m, f = agegrp
        r = conn.acs5.get(('NAME', m, f), {'for':'county:*'})
        df = pd.DataFrame(r)
        dflst.append(df)
    for i, df in enumerate(dflst):
        colname = "total_" + str(i)
        df[colname] = df[df.columns[0]] + df[df.columns[1]]
    mgd = reduce(lambda x, y: pd.merge(x, y, on='NAME'), dflst)
    age_labels = {'total_0': 'under5', 'total_1': '5to9', 'total_2': '10to14',
    'total_3': '15to17', 'total_4': '18to19', 'total_5': '20', 'total_6': '21',
    'total_7': '22to24', 'total_8': '25to29', 'total_9': '30to34',
    'total_10': '35to39', 'total_11': '40to44', 'total_12': '45to49',
    'total_13': '50to54', 'total_14': '55to59', 'total_15': '60to61',
    'total_16': '62to64', 'total_17': '65to66', 'total_18': '67to69',
    'total_19': '70to74', 'total_20': '75to79', 'total_21': '80to84',
    'total_22': '85over'}
    fin = get_percents(mgd, "total", age_labels)
    return fin


def make_agegrp_input(agegrp_lst):
    '''
    Getting a range of columns to be pulled from the ACS "sex by age" table,
    which can then be passed to an API call. This differs from the general
    get_inputs function because it must account for pulling two columns at a
    time - the male column and the corresponding female column for a single
    age group.

    Inputs:
    agegrp_lst: a list to hold the names of the desired columns
    '''
    for i in range(3, 26):
        m = i
        f = i + 24
        if m < 10:
            mstr = 'B01001_00' + str(m) + 'E'
        else:
            mstr = 'B01001_0' + str(m) + 'E'
        fstr = 'B01001_0' + str(f) + 'E'
        agegrp_lst.append((mstr, fstr))


def get_median_age(conn):
    '''
    Gets a dataframe showing the median age of each county.

    Inputs:
    conn: a Census object which connects to the census API

    Returns:
    kp: a dataframe of county-level median age
    '''
    r = conn.acs5.get(('NAME', 'B01002_001E'), {'for':'county:*'})
    df = pd.DataFrame(r)
    df.rename(columns={'B01002_001E': 'median_age'}, inplace=True)
    df["state"] = df["state"].astype("int")
    df["state"] = df["state"].astype("str")
    df["fips"] = df["state"] + df["county"]
    kp = df[['NAME', 'fips', 'county', 'state', 'median_age']]
    return kp


def get_races(conn):
    '''
    Gets a dataframe showing the percentage of county population in each race
    category.

    Inputs:
    conn: a Census object which connects to the census API

    Returns:
    fin: a dataframe of county-level race distribution
    '''
    input_tup = get_inputs('B02001', 2, 9)
    r = conn.acs5.get(input_tup, {'for':'county:*'})
    df = pd.DataFrame(r)
    race_labels = {'B02001_002E': 'white_alone', 'B02001_003E': 'black_alone',
    'B02001_004E': 'native_alone', 'B02001_005E': 'asian_alone',
    'B02001_006E': 'pacific_alone', 'B02001_007E': 'other_alone',
    'B02001_008E': 'two_or_more'}
    fin = get_percents(df, "B02001", race_labels)
    return fin


def get_hispanic(conn):
    '''
    Gets a dataframe showing the percentage of county population that is
    Hispanic (which, in the census, is recorded separately from race).

    Inputs:
    conn: a Census object which connects to the census API

    Returns:
    kp: a dataframe of county-level Hispanic population percentage
    '''
    r = conn.acs5.get(('NAME', 'B01001I_001E'), {'for':'county:*'})
    df = pd.DataFrame(r)
    df.rename(columns={'B01001I_001E': 'hispanic'}, inplace=True)
    df["state"] = df["state"].astype("int")
    df["state"] = df["state"].astype("str")
    df["fips"] = df["state"] + df["county"]
    kp = df[['NAME', 'fips', 'county', 'state', 'hispanic']]
    return kp


def get_median_inc(conn):
    '''
    Gets a dataframe showing the median household income in each county.

    Inputs:
    conn: a Census object which connects to the census API

    Returns:
    kp: a dataframe of county-level median income
    '''
    r = conn.acs5.get(('NAME', 'B19013_001E'), {'for':'county:*'})
    df = pd.DataFrame(r)
    df.rename(columns={'B19013_001E': 'median_income'}, inplace=True)
    df["state"] = df["state"].astype("int")
    df["state"] = df["state"].astype("str")
    df["fips"] = df["state"] + df["county"]
    kp = df[['NAME', 'fips', 'county', 'state', 'median_income']]
    return kp


def get_incomes(conn):
    '''
    Gets a dataframe showing the income distribution in each county.

    Inputs:
    conn: a Census object which connects to the census API

    Returns:
    fin: a dataframe of county-level income distribution
    '''
    input_tup = get_inputs('B19001', 2, 18)
    r = conn.acs5.get(input_tup, {'for':'county:*'})
    df = pd.DataFrame(r)
    inc_labels = {'B19001_002E': 'under10k', 'B19001_003E': '10k_14999',
    'B19001_004E': '15k_19999', 'B19001_005E': '20k_24999',
    'B19001_006E': '25k_29999', 'B19001_007E': '30k_34999',
    'B19001_008E': '35k_39999', 'B19001_009E': '40k_44999',
    'B19001_010E': '45k_49999', 'B19001_011E': '50k_59999',
    'B19001_012E': '60k_74999', 'B19001_013E': '75k_99999',
    'B19001_014E': '100k_124999', 'B19001_015E': '125k_149999',
    'B19001_016E': '150k_199999', 'B19001_017E': '200kover'}
    fin = get_percents(df, "B19001", inc_labels)
    return fin


def get_sexes(conn):
    '''
    Gets a dataframe showing the sex distribution in each county.

    Inputs:
    conn: a Census object which connects to the census API

    Returns:
    fin: a dataframe of county-level sex distribution
    '''
    r = conn.acs5.get(('NAME', 'B01001_002E', 'B01001_026E'), {'for':'county:*'})
    df = pd.DataFrame(r)
    sex_labels = {'B01001_002E': 'male', 'B01001_026E': 'female'}
    fin = get_percents(df, "B01001", sex_labels)
    return fin
