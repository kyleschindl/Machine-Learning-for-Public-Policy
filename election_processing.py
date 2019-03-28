'''
It is the data-cleaning code for the presidential
elections of 2008, 2012, and 2016
The data comes from MIT Election Data and Science Lab
Source:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ
We download the data in csv file, whose name is 'countypres_2000-2016.csv'
'''

import pandas as pd

filename_raw_data = 'countypres_2000-2016.csv'


def read_data(filename):
    '''
    Process the raw data downloaded from the
    County Presidential Election Returns Database

    Input: filename (str), the filename of raw data

    Output: processed_data(dataframe), the processed pandas dataframe
    '''

    data = pd.read_csv(filename)
    years = [2008, 2012, 2016]
    data = data[data.year.isin(years)]
    invalid_county_names = ['Statewide writein', 'Maine UOCAVA', \
	    'District 99', 'Federal Precinct']
    for index, row in data.iterrows():
        if row.county in invalid_county_names:
            data.drop(index, inplace=True)
    data.party = data.party.fillna("independent")
    data['FIPS'] = data['FIPS'].astype(int)
    data['FIPS'] = data['FIPS'].astype(str)
    data = data.drop('version', axis=1)
    data = data.dropna()
    data['vote_share'] = data.candidatevotes/data.totalvotes
    processed_data = data.groupby(
        ['FIPS', 'county', 'year', 'party']).sum().reset_index()

    return processed_data


def split_by_year(data, year):
    '''
    Split the processed election data by year

    Input: data (dataframe), the processed pandas dataframe
           year (int), the specific year of presidential election

    Output: data_new(dataframe), the election data of specific year
    '''

    data_new = data[data.year == year]

    return data_new


def county_party_voteshare(data):
    '''
    Find the county-level voteshare of each party

    Input: data (dataframe), the processed election data of specific year

    Ouput: pivot (dataframe), the pivot data that shows the party voteshare
           on county-level
    '''

    pivot = pd.pivot_table(
        data, values='vote_share', index=['FIPS', 'county'], columns='party')

    return pivot


def find_eventual_winner(data):
    '''
    Find the winning party by year and county

    Inputs: data (pandas dataframe)

    Output: pandas dataframe
    '''

    eventual_winner = data.loc[data.groupby(
        ['FIPS', 'year']).vote_share.idxmax()][['FIPS', 'year', 'party']]
    eventual_winner = eventual_winner.reset_index(drop=True)

    return eventual_winner


#Processed the raw data and split the data by year
processed = read_data(filename_raw_data)
winner = find_eventual_winner(processed)
data_2008 = split_by_year(processed, 2008)
pivot_2008 = county_party_voteshare(data_2008)
data_2012 = split_by_year(processed, 2012)
pivot_2012 = county_party_voteshare(data_2012)
data_2016 = split_by_year(processed, 2016)
pivot_2016 = county_party_voteshare(data_2016)


#Convert pandas dataframe to csv files
processed.to_csv("county_total.csv", index=False)
winner.to_csv('winner_by_county_by_year.csv', index=False)
data_2008.to_csv("2008_election_county.csv", index=False)
data_2012.to_csv("2012_election_county.csv", index=False)
data_2016.to_csv("2016_election_county.csv", index=False)
pivot_2008.to_csv("2008_county_voteshare_breakdown.csv")
pivot_2012.to_csv("2012_county_voteshare_breakdown.csv")
pivot_2016.to_csv("2016_county_voteshare_breakdown.csv")
