#!/usr/bin/python
# -*- coding: utf8 -*-
import arcpy
import numpy
import sys  

#Overwrite the output if exist
arcpy.env.overwriteOutput = True

reload(sys)  
sys.setdefaultencoding('utf8')

# Adjust folder directory
infolder = r"Z:\Temp\CHIRPS\Daily\Max3"
outfolder = r"Z:\Temp\CHIRPS\Daily\Pct3"
arcpy.env.workspace = infolder
rasters = arcpy.ListRasters()

nmpyrys = []
for i in rasters:
	nmpyrys.append(arcpy.RasterToNumPyArray(i))
a = numpy.array(nmpyrys)

nmpyrys_m = []
for i in nmpyrys:
	m = numpy.where(a[0] == 0, 1, 0)
	am = numpy.ma.MaskedArray(i, mask=m)
	nmpyrys_m.append(am)
a_m = numpy.array(nmpyrys_m)
n_98 = numpy.nanpercentile(a_m, 98., axis=0)

# Based on lower-left and pixel size global CHIRPS data
arcpy.NumPyArrayToRaster(n_98,
					arcpy.Point(-180.0000000,-50.0000015),
					0.050000000745058,
					0.050000000745058,
					0)

out = arcpy.NumPyArrayToRaster(n_98, arcpy.Point(-180.0000000,-50.0000015), x_cell_size=0.050000000745058, y_cell_size=0.050000000745058, value_to_nodata=0)
out_name = "\\".join([outfolder, "n_98.tif"])
out.save(out_name)