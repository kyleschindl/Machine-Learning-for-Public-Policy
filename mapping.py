'''
This file walks through our process of using Folium to create maps
visualizing demographic and election data.
'''

import folium
import pandas as pd


county_geo = r'us-counties.json'


df = pd.read_csv("polling_data_for_mapping.csv")
df['FIPS_Code'] = df['FIPS_Code'].astype(str)


def get_thresholds(column):
    '''
    Returns values at the 0th, 20th, 40th, 60th, 80th, and 100th percentile of a
    variable.

    Inputs:
    column (str): name of a column in the dataframe df

    Returns:
    A list of quantile values that can be used as threshold values in making
    a map.
    '''
    rv = []
    for i in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        threshold = df[column].quantile(i)
        rv.append(threshold)
    return rv


'''
We referred to these links in the selection of our color schemes:
https://github.com/dsc/colorbrewer-python
http://bl.ocks.org/mhkeller/10504471
'''


def make_map_layer(map_obj, column, color_scheme, thresholds, opacity):
    '''
    Makes a single choropleth map layer using Folium.

    Inputs:
    map_obj: a Folium map object
    column (str): name of a column in the dataframe df, the variable that will
    be displayed on the map
    outname (str): name for output HTML file
    color_scheme (str): name of a ColorBrewer palette to set the map's color
    scheme
    thresholds (list): numbers defining the bins which will be used to assign
    colors to values
    opacity (float): a float setting the opacity of the map layer
    '''
    map_obj.geo_json(geo_path=county_geo, data_out='data1.json', data=df,
                   columns=['FIPS_Code', column],
                   key_on='feature.id',
                   fill_color=color_scheme, fill_opacity=opacity,
                   line_opacity=1.0, threshold_scale=thresholds)


def make_map(columns, outname, color_scheme, thresholds=[0, 20, 40, 60, 80, 100],
             opacity=1.0):
    '''
    Creates a base map using Folium and adds choropleth layers as specified.

    Inputs:
    columns (list): list of column names in the dataframe df; the variables to
    be displayed on the map
    outname (str): name for output HTML file
    color_scheme (str): name of a ColorBrewer palette to set the map's color
    scheme
    thresholds (list): numbers defining the bins which will be used to assign
    colors to values
    opacity (float): a float setting the opacity of the map layers
    '''
    outfile = 'redone_maps/' + outname + '.html'
    map1 = folium.Map(location=[39.8282, -98.5795], zoom_start=5)
    for column in columns:
        make_map_layer(map1, column, color_scheme, thresholds, opacity)
    folium.LayerControl().add_to(map1)
    map1.create_map(path=outfile)


'''
We make the maps of demographic data:

- the percentage of each county's population that is black
- the percentage of each county's population that is Hispanic
- the median household income in each county
- the median age in each county

'''

thresholds = get_thresholds("black_alone")
make_map(["black_alone"], "black_map", "GnBu", thresholds, opacity=0.7)

make_map(["hispanic"], "hispanic_map", "YlGn", opacity=0.7)

make_map(["median_income"], "income_map", "BuGn", [0, 20000, 40000, 60000,
         80000, 100000], opacity=0.7)

thresholds = get_thresholds("median_age")
make_map(["median_age"], "age_map", "RdPu", thresholds, opacity=0.7)


'''
We make maps for the percentage of a county's vote that went to the Democratic
candidate in 2008, 2012, and 2016.
'''

make_map(["democrat_2008", "democrat_2012", "democrat_2016"], "democrat", "PuBu")


'''
We make maps for the percentage change in the number of polling locations in a
county, between 2008, 2012, and 2016.
'''

make_map(["polling_loc_pct_change_08/12", "polling_loc_pct_change_12/16",
         "polling_loc_pct_change_08/16"], "polling", "RdBu",
         [-100, -50, 0, 50, 100, 151])


'''
We make maps for the number of voters per polling location in a county, in 2008,
2012, and 2016.
'''

make_map(["voters/polling_locs_2008", "voters/polling_locs_2012",
         "voters/polling_locs_2016"], "voters", "YlOrRd",
         [0, 100, 250, 500, 1000, 2000])


'''
We make maps for the number of polling workers per polling location in a county,
in 2008, 2012, and 2016.
'''

make_map(["workers/locs_2008", "workers/locs_2012", "workers/locs_2016"],
         "workers", "YlGnBu", [0, 2, 4, 6, 8, 10])


'''
We make maps for the percentage of provisional votes counted in a county,
in 2008, 2012, and 2016.
'''

make_map(["provisional_percent_2008", "provisional_percent_2012",
         "provisional_percent_2016"], "provisional", "YlOrBr")


'''
We make maps for the percentage of voters who cast provisional votes in a
county, in 2008, 2012, and 2016.
'''

make_map(["provisional/voters_2008", "provisional/voters_2012",
         "provisional/voters_2016"], "prov_voters", "OrRd",
         [0, 0.2, 0.4, 0.6, 0.8, 5])

'''
All of the created maps can be viewed in a single file at our project website,
linked in our README.
'''
