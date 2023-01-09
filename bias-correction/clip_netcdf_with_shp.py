# -*- coding: utf-8 -*-
"""
NAME
    clip_netcdf_with_shp.py
    Batch clip  NetCDF in a folder with a shapefile
DESCRIPTION
    Input data for this script will be NetCDF files and a shapefile in a folder
    This script can do batch clipping of NetCDF files, and save it in seperate folder
REQUIREMENT
    It required glob, xarray, rasterio and shapely module. So it will work on any machine environment
EXAMPLES
    python clip_netcdf_with_shp.py
NOTES
    Some adjustment are required: a correct path to the input folder.
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
import glob
import xarray as xr
import rasterio
import rasterio.mask
import shapely
import os

# Make sure to replace these with the correct paths
folder_path = '/path/to/folder/with/nc_files'
shapefile_path = '/path/to/shapefile'
output_folder_path = '/path/to/output/folder/'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Load the shapefile
with rasterio.open(shapefile_path) as src:
    shapes = src.shapes()

# Get a list of all the netCDF files in the folder
nc_files = glob.glob(folder_path + '/*.nc')

for nc_file in nc_files:
    # Load the netCDF file
    ds = xr.open_dataset(nc_file)

    # Clip the netCDF file with the shapefile
    with rasterio.open(nc_file) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta.copy()
    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    # Save the clipped netCDF file
    output_file_path = output_folder_path + nc_file.split('/')[-1]
    with rasterio.open(output_file_path, "w", **out_meta) as dest:
        dest.write(out_image)

print('Done!')