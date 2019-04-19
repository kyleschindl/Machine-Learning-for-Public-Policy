'''
This notebook shows how we prepared a csv file matching states to their
census-defined regions and divisions. This is later used to assign
region/division information to every county in our dataset.
The initial file we used was downloaded from the Census website:
https://www2.census.gov/programs-surveys/popest/geographies/2017/state-geocodes-v2017.xlsx
It was saved as "regioncodes.csv", also available in the directory.
'''

import pandas as pd

df = pd.read_csv("regioncodes.csv")

df["clean_region"] = df.Name.str.extract("(.*)\s+Region", expand=False)
df.clean_region.fillna(method="ffill", axis=0, inplace=True)
df["clean_division"] = df.Name.str.extract("(.*)\s+Division", expand=False)
df.clean_division.fillna(method="ffill", axis=0, inplace=True)
df[["Name", "clean_region", "clean_division", "State (FIPS)"]]
df = df[df["State (FIPS)"] != 0]
df = df[["Name", "State (FIPS)", "clean_region", "clean_division"]]
df.to_csv("state_regions.csv", index=False)
