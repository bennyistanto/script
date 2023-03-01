#!/usr/bin/python
"""
NAME
    csv_concat.py
    Concatenate csv files by column, and use the date on filename as column name
DESCRIPTION
    Input data for this script will use any csv, which must have lon, lat and value (no header)
    All files required to follow the naming convention "_yyyymmdd.csv"
USAGE
    python csv_concat.py <csv_dir>
CONTACT
    Benny Istanto
    Climate Geographer
    GOST, The World Bank
LICENSE
    This script is in the public domain, free from copyrights or restrictions.
VERSION
    $Id$
TODO
    xx
"""
import pandas as pd
import os
from tqdm import tqdm

# Set the path to the folder containing the CSV files
folder_path = os.path.join(os.getcwd(), 'csv')

# Get a list of all the CSV files in the folder
file_list = os.listdir(folder_path)
file_list = [f for f in file_list if f.endswith('.csv')]

# Create an empty dataframe to hold the concatenated data
concat_df = pd.DataFrame(columns=['lon', 'lat'])

# Loop through each file and concatenate its data to the main dataframe
for file_name in tqdm(file_list, desc='Processing files'):
    file_path = os.path.join(folder_path, file_name)
    # naming convention uga_cli_tas_terraclimate_yyyymmdd.csv
    file_date = file_name[25:33]  # Extract the date from the file name
    file_df = pd.read_csv(file_path, header=None, names=['lon', 'lat', file_date])
    concat_df = pd.merge(concat_df, file_df, on=['lon', 'lat'], how='outer')

# Write the concatenated dataframe to a new CSV file
output_file = '../tas.csv'
concat_df.to_csv(output_file, index=False)
