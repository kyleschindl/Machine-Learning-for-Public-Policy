'''
It includes the codes that produce the barplots and line plots
'''

import math
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#the file 'final_counties.csv' contains the joined dataframe of
#demographic data, election data, and
#polling station data on the county-level.
full_data = pd.read_csv('polling_data.csv')
stats_data = pd.read_csv("polling_data_for_stats.csv")


def find_polling_stations_change_by_region(region, year):
    '''
    Create the dataframe that shows the change of polling station
    number by region

    Inputs:
        region: (str), the column of the dataframe that contains
        the geographical unit,
        which is either 'clean_region' (Midwest, Northeast, etc.)
        or 'clean_division'
        (East North Central, East South Central, etc.)

        year: (str), the difference of years, such as '08/12', or'12/16'

    Outputs: polling_diff (dataframe), the pandas dataframe
    that shows the change of polling stations by region
    '''

    buffer_ = 'polling_loc_diff_'
    buffer_full = buffer_ + year
    polling_diff = full_data.groupby([region])[[buffer_full]].sum()

    return polling_diff


def find_polling_stations_density_by_region(region, year):
    '''
    Create the dataframe that shows the average
    polling station density in a specific year

    Inputs:
        region: (str) the column of the dataframe that contains the
        geographical unit,
        which is either 'clean_region' (Midwest, Northeast, etc.)
        or 'clean_division'
        (East North Central, East South Central, etc.)

        year: (int), the specific year

    Outputs: polling (dataframe), the pandas dataframe that shows the average
            polling station density (voters per polling station)
            in a specific year
    '''

    buffer_ = 'voters/polling_locs_'
    buffer_full = buffer_ + str(year)
    polling = full_data.groupby([region])[[buffer_full]].mean()

    return polling


def create_plot_on_polling_station_density(region, filename):
    '''
    Create the line plot that shows the change of average polling station
    density by region

    Inputs:
        region: (str) the column of the dataframe that contains
        the geographical unit,
        which is either 'clean_region' (Midwest, Northeast, etc.)
        or 'clean_division'
        (East North Central, East South Central, etc.)

        filename: (str) the output filename of the line plot
    '''

    stations_by_region_08 = find_polling_stations_density_by_region(
        region, 2008)
    stations_by_region_12 = find_polling_stations_density_by_region(
        region, 2012)
    stations_by_region_16 = find_polling_stations_density_by_region(
        region, 2016)

    stations_total = stations_by_region_08.join(
        stations_by_region_12, how='outer')
    stations_total = stations_total.join(stations_by_region_16, how='outer')
    stations_total = stations_total.reset_index()
    for i in range(1, 4):
        stations_total[stations_total.columns[i]] = \
            stations_total[stations_total.columns[i]].apply(
                lambda x: math.log(x))
    new_stations_total = pd.melt(stations_total, \
    	          id_vars=region, value_vars=stations_total.columns[1:])
    new_stations_total['year'] = new_stations_total['variable'].apply(
        lambda x: x[-4:])
    new_stations_total = new_stations_total.rename(
        columns={'value': 'log(average_polling_station_density)', \
                  region: 'region'})
    sns.set()
    plt1 = plt.figure()
    sns_plot = sns.lineplot(x='year', \
        y='log(average_polling_station_density)', \
        hue='region', data=new_stations_total)
    sns_plot.legend(loc='center right', bbox_to_anchor=(1.25, 0.67), \
        ncol=1, fontsize='xx-small')
    plt1.tight_layout()

    plt1.savefig(filename)


def create_barplot(region, year, filename):
    '''
    Create the barplot that shows the change of polling station number

    Inputs:
        region: (str) the column of the dataframe that
        contains the geographical unit,
        which is either 'clean_region' (Midwest, Northeast, etc.)
        or 'clean_division'
        (East North Central, East South Central, etc.)

        year: (str), the difference of years, such as '08/12', or'12/16'

        filename: (str) the output filename of the line plot
	'''

    barplot_data = find_polling_stations_change_by_region(region, year)
    barplot_data = barplot_data.reset_index()
    barplot_data = barplot_data.rename(columns={region: "region"})
    sns.set()
    plt2 = plt.figure(figsize=(12, 6))
    sns_plot_2 = sns.barplot(
        x='region', y=barplot_data.columns[1], data=barplot_data)
    sns_plot_2.set_xticklabels(
        sns_plot_2.get_xticklabels(), rotation=40, ha="right")
    plt2.tight_layout()

    plt2.savefig(filename)

def create_heatmap(dataframe, corr_columns, size, filename):
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

    corr_matrix = dataframe[corr_columns].corr().fillna(0)
    x, y = size
    png, ax = plt.subplots(figsize=(x,y))

    ax = sns.heatmap(corr_matrix, center=0, cmap=sns.diverging_palette(250, 10,
                                                                  as_cmap=True))

    png.tight_layout()
    png.savefig(filename)


#Create line plots that show the change of average polling station density
#by region
create_plot_on_polling_station_density(
    'clean_region', 'polling_station_density_variation.png')
create_plot_on_polling_station_density(
    'clean_division', 'polling_station_density_variation_by_division.png')


#Create barplots that show the change of polling station number by region
create_barplot('clean_region', '08/12', 'polling_station_change_0812.png')
create_barplot('clean_region', '12/16', 'polling_station_change_1216.png')
create_barplot(
    'clean_division', '08/12', 'polling_station_change_by_division_0812.png')
create_barplot(
    'clean_division', '12/16', 'polling_station_change_by_division_1216.png')


#Create heatmap of correlations from columns of interest in 2008
corr_columns_2008 = ['polling_locations_2008', 'poll_workers_2008', 
                     'polling_loc_pct_change_08_12',
                     'poll_worker_pct_change_08_12', 'absentee_percent_2008',
                     'provisional_percent_2008', 'voters_polling_locs_2008',
                     'workers_locs_2008', 'provisional_voters_2008',
                     'white_alone', 'black_alone', 'native_alone',
                     'asian_alone', 'pacific_alone', 'other_alone',
                     'two_or_more', 'hispanic', 'median_income',
                     'republican_2008']

create_heatmap(stats_data, corr_columns_2008, (11, 9), "heatmap.png")