'''
In this file we will clean various csv files and merge the resulting
dataframes for the purpose of later analyzing the results.
'''

from functools import reduce
import pandas as pd
import numpy as np

'''
We begin by cleaning the Election Administration and Voting Survey (EAVS)
Datasets. Both the 2008 and 2012 dataset consist of several csv files, labeled
A-F while the 2016 dataset contains a single csv file. Because the naming
conventions of each spreadsheet are inconsistent, we construct dictionaries
mapping a given year to its corresponding csv (and the relevant columns to
import) as well as what the column names should be renamed to.
'''

COLUMNS = {"2008": {"A": ['FIPS_CODE', 'JurisName', 'STATE_', "A1"],
                    "C": ['FIPS_CODE', 'JurisName', 'STATE_', 'C1a', 'C1b'],
                    "D": ['FIPS_CODE', 'JurisName', 'STATE_', 'D2a', 'D3'],
                    "E": ['FIPS_CODE', 'JurisName', 'STATE_', 'E1', 'E2a'],
                    "F": ['FIPS_CODE', 'JurisName', 'STATE_', 'F1a']},
           "2012": {"A": ['FIPSCode', 'Jurisdiction', 'State', 'QA1a'],
                    "C": ['FIPSCode', 'Jurisdiction', 'State', 'QC1a', 'QC1b'],
                    "D": ['FIPSCode', 'Jurisdiction', 'State', 'QD2a', 'QD3a'],
                    "E": ['FIPSCode', 'Jurisdiction', 'State', 'QE1a', 'QE1b'],
                    "F": ['FIPSCode', 'Jurisdiction', 'State', 'QF1a']},
           "2016": {"Z": ["FIPSCode", "State", "JurisdictionName", "A1a", "C1a",
                          "C1b", "D2a", "D3a", "E1a", "E1b", "F1a"]}}

COL_NAMES = {"FIPS_CODE": "fips", "JurisName": "jurisdiction",
             "STATE_": "state", "A1": "registered_voters", "D3": "poll_workers",
             "E1": "provisional_sent", "E2a": "provisional_counted",
             "Jurisdiction": "jurisdiction", "QA1a": "registered_voters",
             "QC1a": "absentee_sent", "QC1b": "absentee_counted",
             "QD2a": "polling_locations", "QD3a": "poll_workers",
             "QE1a": "provisional_sent", "QE1b": "provisional_counted",
             "QF1a": "num_voters", "FIPSCode": "fips",
             "JurisdictionName": "jurisdiction", "State": 'state',
             "A1a": "registered_voters", "C1a": "absentee_sent",
             "C1b": "absentee_counted", "D2a": "polling_locations",
             "D3a": "poll_workers", "E1a": "provisional_sent",
             "E1b": "provisional_counted", "F1a": "num_voters"}

COL_TYPES = {'FIPSCode': str, "FIPS_CODE": str}

ELECTIONS = {"2008": ["EAVS/2008_A.csv", "EAVS/2008_C.csv", "EAVS/2008_D.csv",
                      "EAVS/2008_E.csv", "EAVS/2008_F.csv"],
             "2012": ["EAVS/2012_A.csv", "EAVS/2012_C.csv", "EAVS/2012_D.csv",
                      "EAVS/2012_E.csv", "EAVS/2012_F.csv"],
             "2016": ["EAVS/2016_Z.csv"]}

'''
Time invariants are defined as column names that do not change between
election years. When renaming columns based on the election year we check
that columns are not in this list in order to maintain name consistency.
'''

TIME_INVARIANTS = ['FIPS_CODE', 'JurisName', 'STATE_', 'FIPSCode',
                   'Jurisdiction', 'State', 'JurisdictionName']


def clean_csv(filename, cols, col_names, col_types):
    '''
    Reads in a csv file from the EAVS dataset. We then clean the resulting
    dataframe and groupby FIPS codes in order to merge states that incorrectly
    provided city data instead of county data.

    Inputs:
        filename (str): a csv file
        cols (list):columns to import
        col_names (dict): dictionary to rename column names after reading csv
        col_types (dict): dictionary to read in certain column types

    Returns:
        county_groups (dataframe)
    '''

    eav = pd.read_csv(filename, usecols=cols, dtype=col_types,
                      encoding='cp1252')

    #replace empty spaces with NaN
    eav = eav.replace(r'^\s+$', np.nan, regex=True)

    eav = eav.rename(columns=col_names)
    eav = convert_numerics(eav, eav.columns[3:])
    eav['fips'] = eav['fips'].astype(str)

    #Adds a leading zero erroneously removed in the original EAVS data
    eav['fips'] = eav['fips'].apply(lambda x: add_zero_to_FIPS(x, 9))

    #Removes Wisconsin from the dataframe as their data was too inconsistent to
    #include in the dataframe
    eav = eav[eav['state'] != "WI"]

    #Remove superfluous city data from VA data
    eav = eav[~((eav['state'] == "VA") &
                (~eav['jurisdiction'].str.contains("COUNTY")))]

    #Groups by the first five characters of the FIPS code, representing counties
    county_groups = eav.groupby(eav.fips.str[:5]).sum()

    return county_groups


def import_elections(elections):
    '''
    Iterates through the 11 EAVS csv files and cleans each in turn. After
    cleaning, merges the respective dataframes into one final dataframe
    containing all EAVS data for 2008, 2012, and 2016.

    Inputs:
        elections (list): a list of EAVS csv files

    Returns:
        eavs_df (dataframe)

    '''
    elections_dfs = []

    #Iterates over keys: 2008, 2012, 2016
    for election in elections:

        #Iterates over each file associated with a given election year
        for file in elections[election]:

            #file[-5] refers to the character associated with an election's
            #csv label (A, C, etc.)
            cols = COLUMNS[election][file[-5]]

            names = {col: COL_NAMES[col] + "_" + election if col not in
                          TIME_INVARIANTS else COL_NAMES[col] for col in cols}
            types = {col: COL_TYPES[col] for col in cols if col in COL_TYPES}

            elections_dfs.append(clean_csv(file, cols, names, types))

    #Merges the 11 dataframes produced to one single dataframe
    eavs_df = reduce(lambda x, y: pd.merge(x, y, on='fips'), elections_dfs)

    #Sets missing data (signified by a value of -999999 to 0)
    eavs_df[eavs_df < 0] = 0

    return eavs_df


def convert_numerics(df, columns):
    '''
    Takes a dataframe and list of column names, and converts them to numeric
    data types, coercing errors to NaNs.

    Inputs:
        df (dataframe): a dataframe
        columns (list): a list of strings of column names

    Returns:
        df (dataframe)
    '''
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def add_zero_to_FIPS(FIPS, length):
    '''
    Add leading zero for the counties with length-digit FIPS codes

    Inputs:
        FIPS (str): a FIPS code
        length (int): desired length of FIPS code to match

    Returns:
        FIPS (str)
    '''

    if len(FIPS) == length:
        FIPS = '0' + FIPS

    return FIPS


def create_col_names(name):
    '''
    Creates a list of column names for ease of creating new columns
    for each year

    Inputs:
        name (str): desired base of new column name

    Returns:
        a list of strings
    '''

    return [name + year for year in ['2008', '2012', '2016']]


def create_percentages(df, new_name, numerator, denominator, scale):
    '''
    Creates a new column as a percentage of the stated numerator
    and denominator, on the condition that the denominator is not
    equal to zero. Takes scale as a parameter in order to create
    a ratio instead of percentage if desired.

    Inputs:
        df (dataframe): EAVS dataframe
        new_name (str): name of the desired new column
        numerator (str): column name of numerator
        denominator (str): column name of denominator
        scale (int): 100 or 1 depending on if percentage or ratio

    Returns:
        None
    '''

    for col in create_col_names(new_name):

        #col[-4:] represents the given year, e.g. 2008
        year = col[-4:]

        df[col] = np.where(df[denominator + year] != 0,
                           (df[numerator + year] / df[denominator + year])
                           * scale, 0.0)


def find_change(df, name, years):
    '''
    Finds the raw difference between two columns as well as the
    percentage change, on the condition that both columns
    are non-zero in the case of differences, and that the
    denominator is non-zero in the case of percentages.

    Inputs:
        df (dataframe): an EAVS dataframe
        name (str): base name of new column
        years (list): list of strings of column names

    Returns:
        None
    '''

    #Differences
    df[name + "diff_08/12"] = np.where((df[years[0]] > 0) &
                                       (df[years[1]] > 0), df[years[1]] -
                                       df[years[0]], 0.0)

    df[name + "diff_12/16"] = np.where((df[years[1]] > 0) &
                                       (df[years[2]] > 0), df[years[2]] -
                                       df[years[1]], 0.0)

    df[name + "diff_08/16"] = np.where((df[years[0]] > 0) &
                                       (df[years[2]] > 0), df[years[2]] -
                                       df[years[0]], 0.0)

    #Percentages
    df[name + "pct_change_08/12"] = np.where((df[years[0]] > 0),
                                             (df[name + "diff_08/12"] /
                                              df[years[0]]) * 100, 0.0)

    df[name + "pct_change_12/16"] = np.where((df[years[1]] > 0),
                                             (df[name + "diff_12/16"] /
                                              df[years[1]]) * 100, 0.0)

    df[name + "pct_change_08/16"] = np.where((df[years[0]] > 0),
                                             (df[name + "diff_08/16"] /
                                              df[years[0]]) * 100, 0.0)


'''
Now that we have defined the functions we need to further clean the EAVS
dataset, we import the EAVS data and begin cleaning.
'''

elections_df = import_elections(ELECTIONS)

find_change(elections_df, "polling_loc_", create_col_names("polling_locations_"))
find_change(elections_df, "poll_worker_", create_col_names("poll_workers_"))
find_change(elections_df, "num_voter_", create_col_names("num_voters_"))

create_percentages(elections_df, "absentee_percent_",
                   'absentee_counted_', 'absentee_sent_', 100)
create_percentages(elections_df, "provisional_percent_",
                   'provisional_counted_', 'provisional_sent_', 100)
create_percentages(elections_df, "provisional/voters_",
                   'provisional_sent_', 'num_voters_', 100)


#Note that we are using a scale of 1 here in order to create ratios,
#and not percentages
create_percentages(elections_df, 'voters/polling_locs_',
                   'num_voters_', 'polling_locations_', 1)
create_percentages(elections_df, 'workers/locs_',
                   'poll_workers_', 'polling_locations_', 1)

'''
Some counties with a small amount of polling locations reported created
extremely large percentage increase from year to year (e.g. increasing
from 1 to 9 polling locations). We arbitrarily cap these percentages
at 150% for the purposes of graphing the changes.
'''
elections_df.loc[elections_df['polling_loc_pct_change_08/12'] > 150,
                 'polling_loc_pct_change_08/12'] = 150
elections_df.loc[elections_df['polling_loc_pct_change_12/16'] > 150,
                 'polling_loc_pct_change_12/16'] = 150
elections_df.loc[elections_df['polling_loc_pct_change_08/16'] > 150,
                 'polling_loc_pct_change_08/16'] = 150

'''
We have now finished cleaning our EAVS dataframe. Next, we begin importing other
csv files with relevant data and joining them together. We will join the
following csv files:

winner_by_county_by_year:
    - the political party that won a county in a given election (e.g. democrat,
      republican)

all_demographics:
    - the demographics of a given county based on ACS data containing race,
      median age, median income, and gender

state_regions:
    - the geographic regions a given state is associated to (e.g. "South",
      "Midwest", etc.)

vote_shares (2008, 2012, 2016):
    - the percentage of the total vote going to each given political party
      (democrat, republican, and independent)

us_county_data:
    - data on the unemployment rate in 2011. This csv is primarily used to make
      sure no counties are missing from the EAVS datframe, when creating maps
      later
'''

#Merging elections_df with vote_shares
vote_shares = pd.read_csv("winner_by_county_by_year.csv",
                          dtype={"FIPS": str})

vote_shares = vote_shares.pivot(index="FIPS", columns="year", values='party')
vote_shares.index = vote_shares.index.map(lambda x: add_zero_to_FIPS(x, 4))
elections_df = vote_shares.merge(elections_df, left_index=True,
                                 right_index=True, how='left')

elections_df = elections_df.rename(columns={2008: "party_2008",
                                            2012: "party_2012",
                                            2016: "party_2016"})
elections_df = elections_df.fillna(0)


#We now create dummy variables determining whether a county voted
#Republican in a given year
elections_df['voted_republican_2008'] = np.where(
    elections_df['party_2008'] == "republican", 1, 0)

elections_df['voted_republican_2012'] = np.where(
    elections_df['party_2012'] == "republican", 1, 0)

elections_df['voted_republican_2016'] = np.where(
    elections_df['party_2016'] == "republican", 1, 0)


#Merging elections_df with demographics
demographics = pd.read_csv("all_demographics_2010-2016.csv", dtype={'fips':str})
demographics['fips'] = demographics['fips'].apply(lambda x:
                                                  add_zero_to_FIPS(x, 4))
demographics = demographics.set_index('fips')
elections_df = elections_df.merge(demographics, left_index=True,
							      right_index=True, how='left')

#Merging elections_df with regions
regions = pd.read_csv("state_regions.csv",
                      usecols=['State (FIPS)', 'clean_region',
                               'clean_division'])

elections_df = elections_df.reset_index().merge(regions, left_on='state',
                                                right_on='State (FIPS)',
                                                how='left').set_index("FIPS")


def read_vote_pcts(vote_pcts):
    '''
    Reads in the vote_share breakdown csv files and merges them into one
    dataframe containing each election year.

    Inputs:
        vote_pcts (list): a list of csv file name (str)

    Returns:
        vote_percents (dataframe)
    '''

    columns = ['FIPS', 'democrat', 'independent', 'republican']
    votes = create_col_names('votes_')

    vote_dfs = []

    for index, vote in enumerate(votes):
        year = vote[-4:]
        vote = pd.read_csv(vote_pcts[index],
                           dtype={"FIPS": str}, usecols=columns)
        vote['FIPS'] = vote['FIPS'].apply(lambda x: add_zero_to_FIPS(x, 4))

        vote = vote.rename(columns={col: col + "_" + year
                                    for col in columns[1:]})
        vote_dfs.append(vote)

    vote_percents = reduce(lambda x, y: pd.merge(x, y, on="FIPS"), vote_dfs)
    vote_percents = vote_percents.set_index("FIPS")
    vote_percents = vote_percents * 100

    return vote_percents


#Merging elections_df with vote percentages
votes = read_vote_pcts(["2008_county_voteshare_breakdown.csv",
                        "2012_county_voteshare_breakdown.csv",
                        "2016_county_voteshare_breakdown.csv"])

elections_df = elections_df.merge(votes, left_index=True,
                                  right_index=True, how='left')


#Merging full_counties with elections_df
full_counties = pd.read_csv("us_county_data.csv",
	                           usecols=['FIPS_Code', 'Unemployment_rate_2011'],
	                           dtype={'FIPS_Code': str})

full_counties = full_counties.set_index("FIPS_Code")
full_counties = full_counties.merge(elections_df, left_index=True,
                                    right_index=True, how='left').fillna(0)

'''
We have now finished merging and cleaning all dataframes and are free to begin
analyzing the data. We will now write two csv files, one used for creating
map-based visualizations and one for statistical analysis that would be
sensitive to missing data
'''

def remove_missing_states(dataframe, columns):
    '''
    Removes states that did not report certain columns in all three years.

    Inputs:
        dataframe (dataframe): a dataframe
        columns (list): a list of column names

    Returns:
        df (dataframe)
    '''

    for col in columns:
        dataframe = dataframe[dataframe[col] != 0]

    return dataframe


full_states = remove_missing_states(elections_df,
                                    ['polling_loc_pct_change_08/12',
                                     'polling_loc_pct_change_12/16',
                                     'polling_loc_pct_change_08/16'])

full_states.columns = full_states.columns.str.replace("/", "_")
full_states.to_csv("polling_data_for_stats.csv")
elections_df.to_csv("polling_data.csv")


def tidy(df):
    '''
    Cleans a dataframe for panel data analysis. Essentially takes data that
    is "wide", and condenses the data where each row is an observation in
    time for a given county.

    Inputs:
        df (a dataframe)

    Returns:
        A dataframe
    '''
    dfs = []
    for i, years in enumerate(["2008", "2012", "2016"]): 
        new = pd.DataFrame(df.index) 
        new['year'] = years 
        for col in df.columns: 
            if years in col or (i == 0 and "2010" in col):
                name = col[:-5] 
                new[name] = df[col].values 
        dfs.append(new) 
    return reduce(lambda x, y: pd.merge(x, y, how='outer'),
                  dfs).sort_values(by=["FIPS_Code", "year"])


def calc_changes(df, columns):
    '''
    Finds the percent change within a given group for tidied panel data

    Inputs:
        df (a dataframe)
        columns: a list of column names

    Returns:
        None
    '''

    for col in columns:
        df[col + "_pct_change"] = (df.groupby("FIPS_Code")[col]
                                     .apply(lambda x: x.pct_change()))


full_counties = pd.merge(full_counties, elections_df[['clean_region']])

#Replace / in column names for use in statistical modeling
full_counties.columns = full_counties.columns.str.replace("/", "_")
full_counties = tidy(full_counties)

calc_changes(full_counties, ['registered_voters', 'absentee_sent',
            'absentee_counted', 'polling_locations', 'poll_workers',
            'provisional_sent', 'provisional_counted', 'num_voters',
            'provisional_voters', 'voters_polling_locs', 'workers_locs',
            'median_age', 'white_alone', 'black_alone', 'native_alone',
            'asian_alone', 'pacific_alone', 'other_alone', 'two_or_more',
            'hispanic', 'median_income', 'total_popn', 'democrat',
            'independent', 'republican'])

#Replace inf and -1 values with NaN
full_counties = full_counties.replace({-1: np.nan, np.inf: np.nan,
                                       -np.inf: np.nan})

full_counties["polling_locations_diff"] = (full_counties.groupby("FIPS_Code")
                                ['polling_locations'].apply(lambda x: x.diff()))

full_counties.to_csv("stats.csv", index=False)