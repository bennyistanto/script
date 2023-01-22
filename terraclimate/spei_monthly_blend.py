# -*- coding: utf-8 -*-
"""
NAME
    spei_monthly_blend.py
    Global SPEI-based drought indicators blend.
DESCRIPTION
    Input data for this script will use TerraClimate SPEI monthly data in GeoTIFF format
    This experimental drought blends integrate several SPEI scales into a single product.
    The combines 3-, 6-, 9-, 12-, and 24-month SPEI to estimate the overall drought conditions.
METHOD
    The SPEI values are weighted according to a normalized portion of the inverse of their
    contributing periods. For example, a 3-month SPEI is weighted to 0.453 of the toral SPEI blend.
    To calculate the SPEI blend weighting factor, can follow below procedure:
        1. Sum the total number of contributing months (GOST runs a 3-, 6-, 9-, 12-, and 24-months
        SPEI for a total of 54-months)
        2. Divide the total contributing months by the SPEI analysis period (for a 3-month SPEI,
        54/3 = 18)
        3. Divide the fractional contribution by the sum of all of the fractional contributions
        (ensures that total weights always sum to 1.00)
REQUIREMENT
    ArcGIS must installed before using this script, as it required arcpy module.
EXAMPLES
    C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python spei_monthly_blend.py
NOTES
    This script is designed to work with global TerraClimate SPEI-based product.
    If using other data, some adjustment are required: parsing filename, directory, threshold.
    All TerraClimate data and products are available at s3://wbgdecinternal-ntl/climate/
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
from datetime import datetime
from arcpy.sa import *

# Check out the Spatial Analyst extension
arcpy.CheckOutExtension("spatial")

# To avoid overwriting outputs, change overwriteOutput option to False.
arcpy.env.overwriteOutput = True

# Set the workspace
arcpy.env.workspace = r"C:\path\to\workspace"

# Set the input folders
folder_spei03 = r"C:\path\to\spei03"
folder_spei06 = r"C:\path\to\spei06"
folder_spei09 = r"C:\path\to\spei09"
folder_spei12 = r"C:\path\to\spei12"
folder_spei24 = r"C:\path\to\spei24"

# Set the output folder
output_folder = r"C:\path\to\blend"

# Initialize the list of rasters
rasters_spei03 = []
rasters_spei06 = []
rasters_spei09 = []
rasters_spei12 = []
rasters_spei24 = []

# Iterate through the rasters in folder spei03 and add them to the list
for raster in arcpy.ListRasters("*", "TIF"):
    rasters_spei03.append(folder_spei03 + "\\" + raster)

# Iterate through the rasters in folder spei06 and add them to the list
for raster in arcpy.ListRasters("*", "TIF"):
    rasters_spei06.append(folder_spei06 + "\\" + raster)

# Iterate through the rasters in folder spei06 and add them to the list
for raster in arcpy.ListRasters("*", "TIF"):
    rasters_spei09.append(folder_spei09 + "\\" + raster)

# Iterate through the rasters in folder spei12 and add them to the list
for raster in arcpy.ListRasters("*", "TIF"):
    rasters_spei12.append(folder_spei12 + "\\" + raster)

# Iterate through the rasters in folder spei12 and add them to the list
for raster in arcpy.ListRasters("*", "TIF"):
    rasters_spei24.append(folder_spei24 + "\\" + raster)

# Sort the list of rasters by date
rasters_spei03.sort(key=lambda x: x[-8:])
rasters_spei06.sort(key=lambda x: x[-8:])
rasters_spei09.sort(key=lambda x: x[-8:])
rasters_spei12.sort(key=lambda x: x[-8:])
rasters_spei24.sort(key=lambda x: x[-8:])

# Apply the weight on each raster
spei03_weighted = Times(Raster(rasters_spei03), 0.453)
spei03_weighted.save("spei03_weighted")

spei06_weighted = Times(Raster(rasters_spei06), 0.226)
spei06_weighted.save("spei06_weighted")

spei09_weighted = Times(Raster(rasters_spei09), 0.151)
spei09_weighted.save("spei09_weighted")

spei12_weighted = Times(Raster(rasters_spei12), 0.113)
spei12_weighted.save("spei12_weighted")

spei24_weighted = Times(Raster(rasters_spei24), 0.057)
spei24_weighted.save("spei24_weighted")

# calculate the sum of the weighted rasters
output_raster = output_folder + "\\" + "wld_cli_terraclimate_speiblend_03to24_" + datetime.now().strftime("%Y%m%d") + ".tif"
arcpy.CellStatistics_management(["spei03_weighted","spei06_weighted","spei09_weighted","spei12_weighted","spei24_weighted",], output_raster, "SUM", "DATA")

# Check out the Spatial Analyst extension
arcpy.CheckInExtension("spatial")
