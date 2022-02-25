#!/usr/bin/env python

__author__ = 'rochelle'

# Import system modules
from os import listdir, path
#import numpy as np
#from osgeo import gdal

import arcpy

arcpy.CheckOutExtension("spatial")
arcpy.env.overwriteOutput = 1
if arcpy.GetParameterAsText(0) != '':
    inWks = arcpy.GetParameterAsText(0)
else:
    inWks = r'X:\01_Data\02_IDN\Rasters\Climate\Temperature\MODIS\MOD11C3'
if arcpy.GetParameterAsText(1) != '':
    outWks = arcpy.GetParameterAsText(1)
else:
    outWks = r'X:\01_Data\02_IDN\Rasters\Climate\Temperature\MODIS\MOD11C3\Average'
arcpy.env.workspace = inWks

def calcAverage(dayFile, nightFile, avgFile):
    print "calcAverage: ", dayFile, nightFile
    #an empty array/vector in which to store the different bands
    rasters = []
    #open raster
#    ds = gdal.Open(dayFile)
#    ns = gdal.Open(nightFile)
#    rasters.append(ds.GetRasterBand(0).ReadAsArray())
#    rasters.append(ns.GetRasterBand(0).ReadAsArray())
#    raster_stack = np.dstack(rasters)
#    mean_raster = np.mean(raster_stack, axis=2)
    rasters.append(dayFile)
    rasters.append(nightFile)
    outRaster = arcpy.sa.CellStatistics(rasters, "MEAN")
    # Save the output
    outRaster.save(avgFile)
    print "saved avg in: ", avgFile
    return 0

def matchDayNightFiles(dayPath, nightPath, outPath):
    dayFiles = list(listdir(dayPath))
    nightFiles = set(listdir(nightPath))

    print "Day files: ", dayFiles
    print "Night files: ", nightFiles

    for fl in dayFiles:
        # find matching night file
        d_fl, ext = path.splitext(path.basename(path.normpath(fl)))
        if (ext == '.tif'):
            d_t = d_fl.rpartition('.')
            n_fl = d_t[0] + d_t[1] + '06' + ext
            if (n_fl) in nightFiles:
                avg_fl = path.join(outPath, d_t[0] + d_t[1] + 'avg' + ext)
                dp = path.join(dayPath, d_fl+ext)
                np = path.join(nightPath, n_fl)
                calcAverage(dp, np, avg_fl)
    return 0
    
#idn_cli_MYD11C3.A2002.08.005.06.tif
#idn_cli_MYD11C3.A2002.08.005.01.tif

local_path = r'X:\01_Data\02_IDN\Rasters\Climate\Temperature\MODIS\MOD11C3'
matchDayNightFiles(path.join(local_path, "Day"), path.join(local_path, "Night"), outWks)
