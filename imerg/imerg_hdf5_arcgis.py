#-------------------------------------------------------------------------------------------

'''

arcpy script for importing and geolocating GPM IMGER half hourly

data archived in native HDF5 format at NASA GED DISC

https://gpm1.gesdisc.eosdis.nasa.gov/data/s4pa/GPM_L3/GPM_3IMERGHH.05/

 

Wenli Yang

NASA GES DISC

Wenli.Yang@nasa.gov

301-614-5312

'''

#-------------------------------------------------------------------------------------------

'''

Users only need to change, in the following, the subdataset index,

folder name, and input file name.

'''

 

# define the directory where the input file locates

workspace = 'c:\\Users\\username\\Downloads\\'

 

# define the input HDF5 data file

hdf5file = '3B-HHR.MS.MRG.3IMERG.20141001-S090000-E092959.0540.V05B.HDF5'

 

# define the subdataset to import. In this example, the subdataset is the

# "precipitationCal" parameter which is the 5th subdataset as known to ArcMap.

# Unless you are using a different variable from the dataset you don't need to change this.

subdatasetindex = '5'

 

### You do not have to edit anything below this line. ###

# import packages and setup environment

import arcpy

import os

import fileinput

arcpy.gp.overwriteOutput = True

 

# import and rotate to correct orientation

infile = workspace + hdf5file

scratchfile = infile + '-scratch-' + subdatasetindex

outfile = infile + '-' + subdatasetindex

arcpy.ExtractSubDataset_management(infile, scratchfile + '.bil', subdatasetindex)

arcpy.SetRasterProperties_management(scratchfile+'.bil',"#","1 0","#","1 -9999.9000")

arcpy.Rotate_management(scratchfile + '.bil', outfile + '.bil', -90, '#', '#')

 

# remove the layers from data frame so that only the final

# correct layer will appear in the Table of Content window

mxd=arcpy.mapping.MapDocument("CURRENT")

df=arcpy.mapping.ListDataFrames(mxd)[0]

for layer in arcpy.mapping.ListLayers(mxd, "", df):

   if layer.name.find(hdf5file) > -1:

      arcpy.Delete_management(layer)

 

# define correct metadata and load the correct layer

prj = "GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984,6378137.0,298.257223563]],PRIMEM['Greenwich',0.0],UNIT[Degree',0.0174532925199433],AUTHORITY['EPSG',4326]]"

prjfile = open(outfile + '.prj', 'w')

prjfile.write(prj)

prjfile.close()

for line in fileinput.input(outfile + '.hdr', inplace=1):

   if "ULXMAP" in line:

      line = "ULXMAP         -179.95"

      print line

   elif "ULYMAP" in line:

      line = "ULYMAP         89.95"

      print line

   elif "XDIM" in line:

      line = "XDIM           0.1"

      print line

   elif "YDIM" in line:

      line = "YDIM           0.1"

      print line

   elif "NODATA" in line:

      line = "NODATA        -9999.9000"

      print line

   else:

      print line.replace('\n','')

 

fileinput.close()

 

newLayer=arcpy.MakeRasterLayer_management(outfile+'.bil',outfile,"","","")

 

# remove intermediate scratch files, some of which may not

# be removed if not unlocked by ArcMap.

try:

   arcpy.Delete_management(scratchfile+'.bil')

   arcpy.Delete_management(scratchfile+'.hdr')

   arcpy.Delete_management(scratchfile+'.stx')

   arcpy.Delete_management(scratchfile+'.prj')

   arcpy.Delete_management(scratchfile+'.bil.ovr')

   arcpy.Delete_management(scratchfile+'.bil.xml')

   arcpy.Delete_management(scratchfile+'.bil.aux.xml')

except:

   print 'Some scratch files are not deleted.'