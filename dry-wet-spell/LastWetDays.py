#!/usr/bin/env python
__author__ = 'Rochelle'

def lastWetDay(rasters, outraster, numdays):
    # Import system modules
    import arcpy
    from arcpy import env
    from arcpy.sa import HighestPosition


    # Check out the ArcGIS Spatial Analyst extension license
    if arcpy.CheckExtension("Spatial") == "Available":
        arcpy.CheckOutExtension("Spatial")

    # Set local variables
    inRaster01 = "chirps_crop-v2.0.2015.06.26"
    inRaster02 = "chirps_crop-v2.0.2015.06.27"
    inRaster03 = "chirps_crop-v2.0.2015.06.29"

#    inRasters = arcpy.ListRasters("*", "TIFF")


    # Execute HighestPosition
#    outHighestPosition = HighestPosition([inRaster01, inRaster02, inRaster03])
    counter = 0
    rastersCount = len(rasters)

    #collect X consecutive rasters
    lastXRasters = []
    for i in range(counter,(counter+numdays)):
        lastXRasters.append(rasters[i])

    outHighestPosition = HighestPosition(lastXRasters)
    #for ras in raslist:
    #    outRaster = os.path.join(outfolder2, ras + "_Albers")
    #    arcpy.ProjectRaster_management(ras, outRaster, spatialref2)
    #    print str(ras) + " has been reprojected to: " + str(outRaster)

    # Save the output
    outHighestPosition.save(outraster)

    return 0
