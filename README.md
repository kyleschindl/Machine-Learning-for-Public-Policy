Project examining geographic and demographic trends in closure of polling locations across the United States.

Done as a final project for the UChicago CS 122 class in Winter 2019.

The writeup of our process and findings can be viewed at our [project website](https://lilianhj.github.io/polling-closures/index.html).

## Team members:
* [Lilian Huang](https://github.com/lilianhj/)
* [Peter Li](https://github.com/jizhao94)
* [Kyle Schindl](https://github.com/kyleschindl)
          
## Requirements:
### In Python:
* Census
* us
* folium (0.2.1)
* pandas 
* seaborn
* matplotlib
* numpy
* functools
* statsmodels

Other than Folium, we used the most up-to-date versions of all libraries.
### In R:
* dplyr
* stargazer

## Files containing raw data:
* countypres_2000_2016.csv, which is manually downloaded from [this website](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOQCHQ)
* .xls files in the EAVS folder, which are manually downloaded from the Election Administration and Voting Survey (EAVS) [website](https://www.eac.gov/research-and-data/datasets-codebooks-and-surveys/)
* regioncodes.csv, which is manually downloaded from the Census [website](https://www2.census.gov/programs-surveys/popest/geographies/2017/state-geocodes-v2017.xlsx)
* us_county_data.csv, data on the unemployment rate in 2011, which is manually downloaded from [here](https://gist.githubusercontent.com/wrobstory/5609889/raw/d03fa21d0c88712ab6bcdec0ee8ae682ec9b3c2e/us_county_data.csv). This csv is primarily used to make sure no counties are missing from the EAVS dataframe, when creating maps later
* us-counties.json, a json file that allows us to map all counties in the US, which is manually downloaded from [here](https://gist.githubusercontent.com/wrobstory/5586482/raw/6031540596a4ff6cbfee13a5fc894588422fd3e6/us-counties.json)

We also made use of American Community Survey estimates, but these were accessed through an API rather than being downloaded as csvs. The processed version of this data will be produced by running acs_run.py.

## Scripts and files containing code:

The files should be viewed/run in the following order.

### Data collection and initial data cleaning:
1. Accessing the API of ACS data and subsequent data-cleaning:
    * acs_run.py (runs the functions in acs_base.py to access and download ACS data)
2. Cleaning the data on county-level presidential election outcome:
    * election_processing.py
3. Processes regioncodes.csv to prepare a clean csv file matching states to their census-defined regions and divisions:
    * regions_divisions.py

### Merging multiple datasets into one:
* elections_data_processing.py

### Regressions using merged data:

* statistical_analysis.py
* regression.R

### Data representation and visualization:
1. Data visualization with maps:
    * mapping.py
    * All output maps can be viewed in a single webpage at our [project website](https://lilianhj.github.io/polling-closures/maps/index.html)
2. Visualization with line plots and barplots:
    * summary_stats_and_vis.py
