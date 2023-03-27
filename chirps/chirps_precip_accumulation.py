# -*- coding: utf-8 -*-
"""
NAME
    chirps_precip_accumulation.py
    Calculate running dekad of 1-,3-,6-,9-,12- and 24-month accumulation precipitation using dekad data
DESCRIPTION
    Input data for this script will use https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_dekad/tifs/
REQUIREMENT
    ArcGIS must installed before using this script, as it required arcpy module.
EXAMPLES
    C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python chirps_precip_accumulation.py
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
import arcpy
import os
from arcpy.sa import *

# Check out the ArcGIS Spatial Analyst extension license
arcpy.CheckOutExtension("Spatial")

# Input folder containing the CHIRPS GeoTIFF files
input_folder = r"C:\path\to\your\input_folder"

# Output folders for the different accumulations
output_folders = {
    "Month1": r"C:\path\to\your\output_folder\Month1",
    "Month3": r"C:\path\to\your\output_folder\Month3",
    "Month6": r"C:\path\to\your\output_folder\Month6",
    "Month9": r"C:\path\to\your\output_folder\Month9",
    "Month12": r"C:\path\to\your\output_folder\Month12",
    "Month24": r"C:\path\to\your\output_folder\Month24",
}

# Create output folders if they don't exist
for folder in output_folders.values():
    if not os.path.exists(folder):
        os.makedirs(folder)

# Get the list of GeoTIFF files
geotiff_files = [f for f in os.listdir(input_folder) if f.endswith(".tif")]

# Function to accumulate precipitation
def accumulate_precipitation(accumulation, files):
    acc_files = []
    for i in range(len(files) - accumulation + 1):
        last_date = files[i + accumulation - 1].split(".")[1:4]
        last_date = "".join(last_date)
        output_file = os.path.join(output_folders[f"Month{accumulation}"], \
                                  f"wld_cli_precip_{accumulation}months_chirps-v2.0.{last_date}.tif")
        
        # Check if the output file already exists, if so, skip this iteration
        if os.path.exists(output_file):
            print(f"Skipping existing file: {output_file}")
            continue

        acc_rasters = [arcpy.Raster(os.path.join(input_folder, files[j])) for j in range(i, i + accumulation)]
        acc_sum = CellStatistics(acc_rasters, "SUM")
        
        acc_sum.save(output_file)
        acc_files.append(output_file)
    return acc_files

# Calculate accumulations and save them in the corresponding folders
accumulations = [1, 3, 6, 9, 12, 24]
for accumulation in accumulations:
    dekads_per_accumulation = accumulation * 3
    _ = accumulate_precipitation(dekads_per_accumulation, geotiff_files)

print("Completed the precipitation accumulation calculations.")

# Check in the ArcGIS Spatial Analyst extension license
arcpy.CheckInExtension("Spatial")
