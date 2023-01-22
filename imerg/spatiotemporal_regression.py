# -*- coding: utf-8 -*-
"""
NAME
    spatiotemporal_regression.py
DESCRIPTION
    Perform a focal linear regression between two sets of timeseries raster
REQUIREMENT
    It required os, numpy, scipy, rasterio and sklearn module. 
    So it will work on any machine environment.
HOW-TO USE
    python spatiotemporal_regression.py
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
import os
import numpy as np
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
import rasterio
from rasterio.warp import calculate_default_transform, reproject

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
        with rasterio.open(var1_path, "r") as src:
            src_meta = src.meta
            var1_data.append(src.read(1))

        # Read the raster file for variable 2
        var2_path = os.path.join(raster_dir_var2, "hwo_{}{:02d}.tif".format(year, month))
        with rasterio.open(var2_path, "r") as src:
            # Set the CRS to EPSG:4326
            src_meta.update(crs={'init': 'epsg:4326'})
            # Check if the input rasters have the same CRS
            if src.crs != src_meta['crs']:
                # calculate the default transform to match the first raster
                transform, width, height = calculate_default_transform(src.crs, src_meta['crs'], *src.shape)
                # reproject the raster to match the first raster
                var2_data.append(reproject(src.read(1), transform, 'bilinear', width=width, height=height))
            else:
                var2_data.append(src.read(1))

    # Stack the raster data for each variable into a 3D array
    var1_data = np.stack(var1_data)
    var2_data = np.stack(var2_data)

    # Get the shape of the raster data
    nrows, ncols, ntimesteps = var1_data.shape

    # Set the window size to consider 8 pixels around the current pixel
    window_size = 8

    # Initialize arrays to store the slope and intercept for each pixel
    slope = np.empty((nrows, ncols))
    intercept = np.empty((nrows, ncols))

    # Iterate over the rows and columns of the raster data
    for row in range(window_size, nrows-window_size):
        for col in range(window_size, ncols-window_size):
            # Get the raster data for the current pixel and the surrounding pixels
            var1_window = var1_data[row-window_size:row+window_size+1, col-window_size:col+window_size+1, :]
            var2_window = var2_data[row-window_size:row+window_size+1, col-window_size:col+window_size+1, :]

            # Flatten the raster data into a 2D array
            var1_window_flat = var1_window.flatten()
            var2_window_flat = var2_window.flatten()

            # Add a column of ones to the variable data for the intercept term
            X = np.column_stack((var1_window_flat, np.ones(var1_window_flat.shape[0])))

            # Fit a linear regression model to the data
            slope[row, col], intercept[row, col], _, _, _ = stats.linregress(X, var2_window_flat)

    # Write the slope and intercept raster files
    with rasterio.open(os.path.join(output_dir, "slope_{}.tif".format(month)), "w", driver="GTiff",
                       width=ncols, height=nrows, count=1, dtype=np.float32, **meta) as dst:
        dst.write(slope, 1)

    with rasterio.open(os.path.join(output_dir, "intercept_{}.tif".format(month)), "w", driver="GTiff",
                       width=ncols, height=nrows, count=1, dtype=np.float32, **meta) as dst:
        dst.write(intercept, 1)
