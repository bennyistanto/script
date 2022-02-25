#!/usr/bin/python
# -*- coding: utf8 -*-
import arcpy
import os

#Overwrite the output if exist
arcpy.env.overwriteOutput = True

# Set local variables
ws_in = r"Z:\Temp\CHIRPS\SPI\IDN_Month1D1"
ws_out = r"Z:\Temp\CHIRPS\SPI\IDN_Month1D1\Outputs"

#CHIRPS geotiff to nc
variable = "precip"
units = ""
XDimension = "x"
YDimension = "y"
bandDimension = ""

arcpy.env.workspace = ws_in
rasters = arcpy.ListRasters("*","TIF")

for rasname in rasters:
    raspath = os.path.join(ws_in, rasname)
    name, ext = os.path.splitext(rasname)
    netcdf = os.path.join(ws_out, name)

    # Process: RasterToNetCDF
    print "Exporting {0} to {1}".format(rasname, netcdf)
    arcpy.RasterToNetCDF_md(raspath, netcdf, variable, units, XDimension,
                             YDimension, bandDimension)