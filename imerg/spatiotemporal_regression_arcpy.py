# -*- coding: utf-8 -*-
"""
NAME
    spatiotemporal_regression_arcpy.py
DESCRIPTION
    Perform a focal linear regression between two sets of timeseries raster
REQUIREMENT
    ArcGIS must installed before using this script, as it required arcpy module.
EXAMPLES
    \\arcgispro-py3\\python spatiotemporal_regression_arcpy.py
NOTES
    This code performs a linear regression analysis to identify the relationship between 
    two variables (var1 and var2) represented in two sets of raster files. It reads in 
    raster files for var1 and var2 for each month and year between 2001 and 2022, and 
    then performs a linear regression analysis on a window of 8 pixels around each pixel 
    in the images. It then saves the slope and intercept values of the linear regression 
    model to output raster files.
DATA
    IMERG: https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.06/
    Maximum daily rainfall in a month, 2000 - 2021
    Global Surface Water: https://global-surface-water.appspot.com/download
    Historical water occurrence derived from Monthly Water History
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
import arcpy
from arcpy.sa import *
from arcpy.mp import LinearRegression

# Set the path to the directory containing the raster files
raster_dir_var1 = "path/to/raster/folder_var1"
raster_dir_var2 = "path/to/raster/folder_var2"
output_dir = "path/to/output"

# Iterate through the months
for month in range(1, 13):
    # Initialize lists to store the raster data for each variable
    var1_data = []
    var2_data = []
    # Iterate through the years
    for year in range(2000, 2022):
        # Read the raster file for variable 1
        var1_path = os.path.join(raster_dir_var1, "imerg_{}{:02d}.tif".format(year, month))
        var1_data.append(arcpy.Raster(var1_path))

        # Read the raster file for variable 2
        var2_path = os.path.join(raster_dir_var2, "hwo_{}{:02d}.tif".format(year, month))
        with arcpy.Describe(var2_path) as desc:
            # Check if the input rasters have the same CRS
            if desc.spatialReference.name != "GCS_WGS_1984":
                # Reproject the raster to match the first raster
                var2_data.append(ProjectRaster(var2_path, var1_path, "GCS_WGS_1984"))
            else:
                var2_data.append(arcpy.Raster(var2_path))

    # Stack the raster data for each variable into a 3D array
    var1_data = Con(IsNull(var1_data), -9999, var1_data)
    var2_data = Con(IsNull(var2_data), -9999, var2_data)

    # Get the shape of the raster data
    nrows = var1_data.height
    ncols = var1_data.width
    ntimesteps = len(var1_data)

    # Set the window size to consider 8 pixels around the current pixel
    window_size = 8

    # Initialize arrays to store the slope and intercept for each pixel
    slope = arcpy.NumPyArray(nrows, ncols)
    intercept = arcpy.NumPyArray(nrows, ncols)

    # Iterate over the rows and columns of the raster data
    for row in range(window_size, nrows-window_size):
        for col in range(window_size, ncols-window_size):
            # Get the raster data for the current pixel and the surrounding pixels
            var1_window = var1_data[row-window_size:row+window_size+1, col-window_size:col+window_size+1, :]
            var2_window = var2_data[row-window_size:row+window_size+1, col-window_size:col+window_size+1, :]

            # Flatten the raster data into a 2D array
            var1_window_flat = arcpy.da.TableToNumPyArray(var1_window, ['Value'])['Value']
            var2_window_flat = arcpy.da.TableToNumPyArray(var2_window, ['Value'])['Value']

            # Fit a linear regression model to the data
            regression_result = LinearRegression(var1_window_flat, var2_window_flat)
            slope[row, col] = regression_result.coefficients['var1_window_flat']
            intercept[row, col] = regression_result.coefficients['constant']

    # Write the slope and intercept raster files
    slope.save(os.path.join(output_dir, "slope_{}{:02d}.tif".format(year, month)))
    intercept.save(os.path.join(output_dir, "intercept_{}{:02d}.tif".format(year, month)))
