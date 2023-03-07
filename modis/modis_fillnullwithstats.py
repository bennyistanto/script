# -*- coding: utf-8 -*-
"""
NAME
    modis_fillnullwithstats.py
    Filling null on MXD13Q1 data with long-term mean
DESCRIPTION
    Input data for this script will use MXD13Q1 16-days data generate from GEE or downloaded from NASA
    This script can do filling null on MXD13Q1 data with long-term mean
REQUIREMENT
    ArcGIS must installed before using this script, as it required arcpy module.
EXAMPLES
    C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python modis_fillnullwithstats.py
NOTES
    This script is designed to work with MODIS naming convention
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
from datetime import datetime, timedelta

# To avoid overwriting outputs, change overwriteOutput option to False.
arcpy.env.overwriteOutput = True

# ISO3 Country Code
iso3 = "syr" # Syria

# Define input and output folders
input_folder = "X:\\Temp\\modis\\{}\\gee\\02_positive".format(iso3)
stats_folder = "X:\\Temp\\modis\\{}\\gee\\03_statistics".format(iso3)
fnws_folder = "X:\\Temp\\modis\\{}\\gee\\04_fillnullwithstats".format(iso3)

def fill_null_with_stats(input_folder, stats_folder, fnws_folder):
    # Loop through the files in the input folder
    for raster in os.listdir(input_folder):
        # Check if the file is a TIFF file
        if raster.endswith(".tif") or raster.endswith(".tiff"):
            # Get the year, month, and day or day of year from the raster name
            # syr_phy_mxd13q1_evi_20020101.tif
            if "_" in raster:
                year = int(raster[20:24])
                month = int(raster[24:26])
                day = int(raster[26:28])
                date = datetime(year, month, day)
                doy = date.strftime('%j')
            else:
                year = int(raster[20:24])
                doy = int(raster[24:27])
                leap_year = (datetime(year, 2, 29).strftime('%j') != '059')
                if leap_year:
                    doy += 1
                    date = datetime(year, 1, 1) + timedelta(doy - 1)

            # Get the DOY from the date
            doy = date.strftime('%j')

            # Construct the corresponding filename in the ndvi_stats folder
            stats_raster = "{0}_phy_mxd13q1_20yr_avg_{1}.tif".format(iso3, doy.zfill(3))
            stats_raster = os.path.join(stats_folder, stats_raster)

            # Check if the corresponding raster exists in the stats folder
            if not arcpy.Exists(stats_raster):
                print("Error: {} not found in {}".format(stats_raster, stats_folder))
                continue

            # Create the output raster name with the appropriate format
            if "_" in raster:
                output_raster = os.path.join(fnws_folder, "{}_phy_mxd13q1_evi_{}{:02d}{:02d}.tif".format(iso3, year, month, day))
            else:
                output_raster = os.path.join(fnws_folder, "{}_phy_mxd13q1_evi_{}{}.tif".format(iso3, year, doy.zfill(3)))

            # Fill null values with the corresponding values from the stats raster
            arcpy.CheckOutExtension("spatial")
            in_raster = arcpy.Raster(os.path.join(input_folder, raster))
            stats_raster = arcpy.Raster(stats_raster)
            out_raster = arcpy.sa.Con(arcpy.sa.IsNull(in_raster), stats_raster, in_raster)

            # Save the output raster
            out_raster.save(output_raster)
            print(output_raster + " completed")
            arcpy.CheckInExtension("spatial")

# Main function
def main():
    # Call the fill_null_with_stats() function for the input folder
    fill_null_with_stats(input_folder, stats_folder, fnws_folder)

if __name__ == '__main__':
    main()