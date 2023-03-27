# -*- coding: utf-8 -*-
"""
NAME
    chirps_longterm_stats.py
    CHIRPS precipitation statistics data 1991-2020, long-term average, max, min and stdev
DESCRIPTION
    Input data for this script will use data generated from chirps_precip_accumulation.py
REQUIREMENT
    ArcGIS must installed before using this script, as it required arcpy module.
EXAMPLES
    C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python chirps_longterm_stats.py
NOTES
    This script is designed to work with CHIRPS naming convention
    If using other data, some adjustment are required: parsing filename and directory
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
import os
import arcpy
from collections import defaultdict
from datetime import datetime

# To avoid overwriting outputs, change overwriteOutput option to False.
arcpy.env.overwriteOutput = True

# Change the data and output folder
input_base_folder = r"C:\path\to\your\input_folder"
output_base_folder = r"C:\path\to\your\output_folder"

# Define statistic names
# Statistics type.
    # MEAN — The mean (average) of the inputs will be calculated.
    # MAJORITY — The majority (value that occurs most often) of the inputs will be determined.
    # MAXIMUM — The maximum (largest value) of the inputs will be determined.
    # MEDIAN — The median of the inputs will be calculated. Note: The input must in integers
    # MINIMUM — The minimum (smallest value) of the inputs will be determined.
    # MINORITY — The minority (value that occurs least often) of the inputs will be determined.
    # RANGE — The range (difference between largest and smallest value) of the inputs will be calculated.
    # STD — The standard deviation of the inputs will be calculated.
    # SUM — The sum (total of all values) of the inputs will be calculated.
    # VARIETY — The variety (number of unique values) of the inputs will be calculated.
stat_names = {"MAXIMUM": "max", "MINIMUM": "min", "MEAN": "avg", "MEDIAN": "med", "STD": "std"}

# Calculate statistics for each time scale
time_scales = ['Dekad', 'Month1', 'Month3', 'Month6', 'Month9', 'Month12', 'Month24']
for time_scale in time_scales:
    input_folder = os.path.join(input_base_folder, time_scale)
    output_folder = os.path.join(output_base_folder, f"Statistics_{time_scale}")

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create file collection based on date information
    groups = defaultdict(list)

    for file in os.listdir(input_folder):
        if file.endswith(".tif") or file.endswith(".tiff"):
            # Parsing the filename to get date information
            parts = file.split("_")
            date_str = parts[-1].split(".")[1:4]  # get the year, month, and day part
            date_str = "".join(date_str)
            date = datetime.strptime(date_str, "%Y%m%d")
            
            # Calculate the dekad
            month = date.month
            dekad = date.day

            # Create MMD format
            mmd = f"{month:02d}{dekad:1d}"

            # Filter files based on the year (1991-2020)
            if 1991 <= date.year <= 2020:
                # Add the file to the group for its Month and Dekad (mmd).
                groupkey = mmd
                groups[groupkey].append(os.path.join(input_folder, file))
                
    for groupkey, files in groups.items():
        print(files)

        # Output file extension
        ext = ".tif"

        # Output filenames
        newfilenames = [
            f"wld_cli_chirps_precip_{time_scale}_{stat}_{groupkey}{ext}"
            for stat in stat_names.values()
        ]

        for i, filename in enumerate(newfilenames):
            if arcpy.Exists(os.path.join(output_folder, filename)):
                print(filename + " exists")
            else:
                stat_type = list(stat_names.keys())[list(stat_names.values()).index(list(stat_names.values())[i])]

                arcpy.CheckOutExtension("spatial")
                outCellStatistics = arcpy.sa.CellStatistics(files, stat_type, "DATA")
                outCellStatistics.save(os.path.join(output_folder, filename))
                arcpy.CheckInExtension("spatial")

                print(filename + " completed")
