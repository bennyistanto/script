# -*- coding: utf-8 -*-
"""
NAME
    modis_ratioanom.py
    MXD13Q1 ratio anomaly
DESCRIPTION
    Input data for this script will use MXD13Q1 16-days data generate from GEE or downloaded from NASA
    This script can do ratio anomaly calculation by comparing with 16-days statistics (AVERAGE)
REQUIREMENT
    ArcGIS must installed before using this script, as it required arcpy module.
EXAMPLES
    C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python modis_ratioanom.py
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
from datetime import datetime

# Define input and output folders
ndvi_folder = r"C:\path\to\ndvi_folder"
ndvi_stats_folder = r"C:\path\to\ndvi_stats"
output_folder = r"C:\path\to\output"

def ratio_anomaly_ddd(raster, ndvi_folder, ndvi_stats_folder, output_folder):
    # Get the year and day of year (yyyyddd) from the raster name
    year = int(raster[5:9])
    doy = int(raster[9:12])
    # Check if the year is a leap year
    leap_year = (datetime(year, 2, 29).strftime('%j') != '059')
    if leap_year:
        doy += 1
    # Construct the corresponding filename in the ndvi_stats folder
    stats_raster = "ndvi_stats_{:03d}.tif".format(doy)
    stats_raster = os.path.join(ndvi_stats_folder, stats_raster)
    # Check if the corresponding raster exists in the ndvi_stats folder
    if not arcpy.Exists(stats_raster):
        if leap_year:
            doy -= 2
        else:
            doy -= 1
        stats_raster = "ndvi_stats_{:03d}.tif".format(doy)
        stats_raster = os.path.join(ndvi_stats_folder, stats_raster)
        if not arcpy.Exists(stats_raster):
            print("Error: {} not found in {}".format(stats_raster, ndvi_stats_folder))
            return
    # Create the output raster name
    output_raster = os.path.join(output_folder, "ratio_anomaly_{}.tif".format(doy))
    # Calculate the ratio anomaly
    anomaly = Raster(os.path.join(ndvi_folder, raster)) / Raster(stats_raster)
    # Multiply by 100 to get the percentage
    anomaly = anomaly * 100
    # Save the output raster
    anomaly.save(output_raster)

def ratio_anomaly_yyyy_mm_dd(raster, ndvi_folder, ndvi_stats_folder, output_folder):
    # Get the year, month, and day from the raster name
    year = int(raster[5:9])
    month = int(raster[10:12])
    day = int(raster[13:15])
    # Get the day of year from the date
    doy = datetime(year, month, day).strftime('%j')
    # Construct the corresponding filename in the ndvi_stats folder
    stats_raster = "ndvi_stats_{}.tif".format(doy)
    stats_raster = os.path.join(ndvi_stats_folder, stats_raster)
    # Check if the corresponding raster exists in the ndvi_stats folder
    if not arcpy.Exists(stats_raster):
        print("Error: {} not found in {}".format(stats_raster, ndvi_stats_folder))
        return
    # Create the output raster name
    output_raster = os.path.join(output_folder, "ratio_anomaly_{}.tif".format(doy))
    # Calculate the ratio anomaly
    anomaly = Raster(os.path.join(ndvi_folder, raster)) / Raster(stats_raster)
    # Multiply by 100 to get the percentage
    anomaly = anomaly * 100
    # Save the output raster
    anomaly.save(output_raster)

# Main function
def main():
    # Loop through the rasters in the ndvi_folder
    for raster in arcpy.ListRasters("ndvi_*", "TIF"):
        if "_" in raster:
            ratio_anomaly_yyyy_mm_dd(raster, ndvi_folder, ndvi_stats_folder, output_folder)
        else:
            ratio_anomaly_ddd(raster, ndvi_folder, ndvi_stats_folder, output_folder)

if __name__ == '__main__':
    main()
