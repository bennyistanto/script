# -*- coding: utf-8 -*-
"""
NAME
    imerg_annual.py
    Global IMERG annual statistics
DESCRIPTION
    Input data for this script will use IMERG in GeoTIFF format
    This script can do annual cell statistics to get MEAN, MAX, MIN and STDEV
REQUIREMENT
    ArcGIS must installed before using this script, as it required arcpy module.
EXAMPLES
    C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python imerg_annual.py
NOTES
    This script is designed to work with global IMERG data
    If using other data, some adjustment are required: parsing filename, directory, threshold
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

# Calendar year
year = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', 
        '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']

# Change the data and output folder
input_folder = "X:\\Temp\\imerg\\data\\geotiff\\precipitation_1days"
output_folder = "X:\\Temp\\imerg\\data\\geotiff\\statistics"

dictionary = {}

for i in year:
    content = []

    for file_annual in os.listdir(input_folder):
        
        if file_annual.endswith(".tif") or file_annual.endswith(".tiff"):
            # Parsing the filename to get YYYY information
            i_imerg = file_annual.index('imerg_')
            # 6 is length of 'imerg_', and 4 is length of yyyy
            yyyy = file_annual[i_imerg + 6:i_imerg+6+4]

            if yyyy == i:
                content.append(os.path.join(input_folder, file_annual))
    
    dictionary[i] = content

for index in dictionary:
    listoffile = dictionary[index]
    print(listoffile)

    ext = ".tif"

    # Output filename
    newfilename_annual_max = 'wld_cli_precip_1d_max_imerg_{0}{1}'.format(index, ext)
    newfilename_annual_min = 'wld_cli_precip_1d_min_imerg_{0}{1}'.format(index, ext)
    newfilename_annual_avg = 'wld_cli_precip_1d_avg_imerg_{0}{1}'.format(index, ext)
    newfilename_annual_std = 'wld_cli_precip_1d_std_imerg_{0}{1}'.format(index, ext)
    print(newfilename_annual_max)

    if arcpy.Exists(os.path.join(output_folder, newfilename_annual_max)):
        print(newfilename_annual_max + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_max = arcpy.sa.CellStatistics(listoffile, "MAXIMUM", "DATA")
        outCellStatistics_max.save(os.path.join(output_folder, newfilename_annual_max))
        arcpy.CheckInExtension("spatial")

    if arcpy.Exists(os.path.join(output_folder, newfilename_annual_min)):
        print(newfilename_annual_min + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_min = arcpy.sa.CellStatistics(listoffile, "MINIMUM", "DATA")
        outCellStatistics_min.save(os.path.join(output_folder, newfilename_annual_min))
        arcpy.CheckInExtension("spatial")

    if arcpy.Exists(os.path.join(output_folder, newfilename_annual_avg)):
        print(newfilename_annual_avg + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_avg = arcpy.sa.CellStatistics(listoffile, "MEAN", "DATA")
        outCellStatistics_avg.save(os.path.join(output_folder, newfilename_annual_avg))
        arcpy.CheckInExtension("spatial")

    if arcpy.Exists(os.path.join(output_folder, newfilename_annual_std)):
        print(newfilename_annual_std + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_std = arcpy.sa.CellStatistics(listoffile, "STD", "DATA")
        outCellStatistics_std.save(os.path.join(output_folder, newfilename_annual_std))
        arcpy.CheckInExtension("spatial")