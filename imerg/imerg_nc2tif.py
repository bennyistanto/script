# -*- coding: utf-8 -*-
"""
NAME
    imerg_nc2tif.py
    Global IMERG daily nc4 translation to GeoTIFF
DESCRIPTION
    Input data for this script will use IMERG Final or Late Run downloaded from NASA website
    This script can do translation from nc4 to GeoTIFF and mask the sea.
REQUIREMENT
    ArcGIS must installed before using this script, as it required arcpy module.
EXAMPLES
    C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python imerg_nc2tif.py
NOTES
    This script is designed to work with global IMERG data (Final or Late Run)
    If using other data, some adjustment are required: parsing filename, directory, threshold
    All IMERG data in nc4 format are available at JNB Server Drive J:\\Data\\GLOBAL\\CLIMATE\\imerg\\nc4\\
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
from arcpy.sa import *
import os


# To avoid overwriting outputs, change overwriteOutput option to False.
arcpy.env.overwriteOutput = True


# IMeRG NC4 to TIFF and extract rainfall data only for land area
def execute_nc2tif(nc2tif_temp, nc2tif_final, ncfolder):
    print("start reading list of precipitation data in netCDF format....")
    print("looking for file with naming 3B-DAY.MS.MRG.3IMERG.YYYYMMDD-S000000-E235959.V06.nc4")

    # Check out any necessary licenses.
    arcpy.CheckOutExtension("spatial")
    arcpy.env.workspace = ncfolder

    # Spatial reference WGS-84
    sr = arcpy.SpatialReference(4326)
    land_subset = "X:\\Temp\\imerg\\subset\\wld_bnd_subset_imerg_01_deg_grid_diss_a.shp"

    # Loop through a list of files in the workspace
    ncfiles = arcpy.ListFiles("*.nc4")

    for nc4 in ncfiles:
        inRasterLayer = nc4

        # Parse IMERG file name without extention
        baseName = os.path.basename(inRasterLayer)
        parseName = os.path.splitext(baseName)[0]

        # Output copy raster
        outRasterLayer = os.path.join(nc2tif_temp, parseName+".tif")

        # Create netCDF Raster layer and save into memory
        outNCLayer = "precipitationCal_Layer"
        print("Create netCDF Raster Layer for "+nc4)
        arcpy.md.MakeNetCDFRasterLayer(in_netCDF_file=inRasterLayer, variable="precipitationCal", \
            x_dimension="lon", y_dimension="lat", out_raster_layer=outNCLayer, \
            band_dimension="", dimension_values=[], value_selection_method="BY_VALUE", cell_registration="CENTER")
        print(nc4+" Raster Layer is successfully created")

        # Copy raster
        print("Copy "+nc4+ " from memory to new GeoTIFF")
        arcpy.management.CopyRaster(in_raster=outNCLayer, out_rasterdataset=outRasterLayer, \
            config_keyword="", background_value=None, nodata_value="-3.40282346639e+38", onebit_to_eightbit="NONE", \
            colormap_to_RGB="NONE", pixel_type="32_BIT_FLOAT", scale_pixel_value="NONE", RGB_to_Colormap="NONE", \
            format="TIFF", transform="NONE", process_as_multidimensional="CURRENT_SLICE", \
            build_multidimensional_transpose="NO_TRANSPOSE")
        print("wld_cli_precip_1d_imerg_"+ymd+".tif is successfully created")

        # Parse date information YYYYMMDD
        i_imerg = nc4.index('3IMERG.')
        ymd = nc4[i_imerg + 7:i_imerg+7+8]
        
        # Output clip layer using new filename and add YYYYMMDD information
        outClipLayer = os.path.join(nc2tif_final, "wld_cli_1d_imerg_"+ymd+".tif")

        # Extract by mask
        print("Clip raster "+parseName+".tif with land subset")     
        outExtractByMask = arcpy.sa.ExtractByMask(in_raster=outRasterLayer, in_mask_data=land_subset)
        outExtractByMask.save(os.path.join(nc2tif_final, outClipLayer))
        arcpy.DefineProjection_management(os.path.join(nc2tif_final, outClipLayer),sr)
        print("Clipped "+parseName+".tif is successfully created")

    arcpy.CheckInExtension("spatial")
    print("Translating netCDF to GeoTIFF and clip with land subset is completed")

# Let's go!
if __name__ == '__main__':
    # Global Environment settings
    with arcpy.EnvManager(scratchWorkspace=r"X:\ArcGIS_TEMP\Scratch.gdb", \
        workspace=r"X:\ArcGIS_TEMP\Default.gdb"):
        # Run the function (output folder temporary, output folder final, input folder)
        execute_nc2tif('X:\\Temp\\imerg\\data\\geotiff\\temporary\\original',\
            'X:\\Temp\\imerg\\data\\geotiff\\rainfall_1days','X:\\Temp\\imerg\\data\\nc4\\finalrun')