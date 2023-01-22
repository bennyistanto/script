# -*- coding: utf-8 -*-
"""
NAME
    modis_16daystats.py
    MXD13Q1 16-days statistics data, long-term average, max, min and stdev
DESCRIPTION
    Input data for this script will use MXD13Q1 16-days data generate from GEE or downloaded from NASA
    This script can do 16-days statistics calculation (AVERAGE, MAXIMUM, MINIMUM and STD)
REQUIREMENT
    ArcGIS must installed before using this script, as it required arcpy module.
EXAMPLES
    C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python modis_16daystats.py
NOTES
    This script is designed to work with MODIS naming convention
    If using other data, some adjustment are required: parsing filename and directory
CONTACT
    Benny Istanto
    Climate Geographer
    GOST/DECAT/DEC Data Group, The World Bank
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

# To avoid overwriting outputs, change overwriteOutput option to False.
arcpy.env.overwriteOutput = True

# Change the data and output folder
input_folder = "X:\\Temp\\modis\\ukr\\gee\\02_positive"
output_folder = "X:\\Temp\\modis\\ukr\\gee\\03_statistics\\temp"

# Create file collection based on MM information
groups = defaultdict(list)

for file_16days in os.listdir(input_folder):
    if file_16days.endswith(".tif") or file_16days.endswith(".tiff"):
        # Parsing the filename to get MM information
        i_evi = file_16days.index('evi_')
        # 4+4 is length of 'evi_' and number of character to skip (yyyy),
        # and 4+8 is length of 'evi_' and yyyymmdd
        groupkey = file_16days[i_evi + 4+4:i_evi+4+8]
        fpath = os.path.join(input_folder, file_16days)
        groups[groupkey].append(fpath)

for groupkey, files in groups.items():
    print(files)

    ext = ".tif"

    # Output filename
    newfilename_16days_max = 'ukr_phy_mxd13q1_16days_20yr_max_evi_{0}{1}'.format(groupkey, ext)
    newfilename_16days_min = 'ukr_phy_mxd13q1_16days_20yr_min_evi_{0}{1}'.format(groupkey, ext)
    newfilename_16days_avg = 'ukr_phy_mxd13q1_16days_20yr_avg_evi_{0}{1}'.format(groupkey, ext)
    newfilename_16days_med = 'ukr_phy_mxd13q1_16days_20yr_med_evi_{0}{1}'.format(groupkey, ext)
    newfilename_16days_std = 'ukr_phy_mxd13q1_16days_20yr_std_evi_{0}{1}'.format(groupkey, ext)
    print(newfilename_16days_max)

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


    # To get another stats, you can duplicate 7 lines below and adjust the statistics type.
    # Don't forget to add additional output file name, you can copy from line 60.
    if arcpy.Exists(os.path.join(output_folder, newfilename_16days_max)):
        print(newfilename_16days_max + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_max = arcpy.sa.CellStatistics(files, "MAXIMUM", "DATA")
        outCellStatistics_max.save(os.path.join(output_folder, newfilename_16days_max))
        arcpy.CheckInExtension("spatial")

    if arcpy.Exists(os.path.join(output_folder, newfilename_16days_min)):
        print(newfilename_16days_min + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_min = arcpy.sa.CellStatistics(files, "MINIMUM", "DATA")
        outCellStatistics_min.save(os.path.join(output_folder, newfilename_16days_min))
        arcpy.CheckInExtension("spatial")

    if arcpy.Exists(os.path.join(output_folder, newfilename_16days_avg)):
        print(newfilename_16days_avg + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_avg = arcpy.sa.CellStatistics(files, "MEAN", "DATA")
        outCellStatistics_avg.save(os.path.join(output_folder, newfilename_16days_avg))
        arcpy.CheckInExtension("spatial")

    if arcpy.Exists(os.path.join(output_folder, newfilename_16days_med)):
        print(newfilename_16days_med + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_med = arcpy.sa.CellStatistics(files, "MEDIAN", "DATA")
        outCellStatistics_med.save(os.path.join(output_folder, newfilename_16days_med))
        arcpy.CheckInExtension("spatial")

    if arcpy.Exists(os.path.join(output_folder, newfilename_16days_std)):
        print(newfilename_16days_std + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_std = arcpy.sa.CellStatistics(files, "STD", "DATA")
        outCellStatistics_std.save(os.path.join(output_folder, newfilename_16days_std))
        arcpy.CheckInExtension("spatial")
