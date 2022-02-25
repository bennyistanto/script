# -*- coding: utf-8 -*-
import os
import arcpy

# MODIS 16 days calendar using Julian Date
chirps_16days_data = ['001', '017', '033', '049', '065', '081', '097', '113', '129', '145', '161', '177',
                  '193', '209', '225', '241', '257', '273', '289', '305', '321', '337', '353']

# Change the data and output folder
data_folder = "X:\\01_Data\\01_Global\\Rasters\\Climate\\Precipitation\\CHIRPS\\By16days"
lta_folder = "X:\\01_Data\\01_Global\\Rasters\\Climate\\Precipitation\\CHIRPS\\Statistics_By16days"

dictionary = {}
for i in chirps_16days_data:
    content = []
    for file_16days in os.listdir(data_folder):
        if file_16days.endswith(".tif") or file_16days.endswith(".tiff"):
            parse_string = file_16days.split('.')
            Dmonthseq = parse_string[3]
            if Dmonthseq == i:
                content.append(os.path.join(data_folder, file_16days))
    dictionary[i] = content


for index in dictionary:
    listoffile = dictionary[index]
    print(listoffile)
    ext = ".tif"
    newfilename_16days_std = 'chirps-v2.0.1981-2019.{0}.16days.39yrs.std{1}'.format(index, ext)
    newfilename_16days_avg = 'chirps-v2.0.1981-2019.{0}.16days.39yrs.avg{1}'.format(index, ext)
    newfilename_16days_max = 'chirps-v2.0.1981-2019.{0}.16days.39yrs.max{1}'.format(index, ext)
    newfilename_16days_min = 'chirps-v2.0.1981-2019.{0}.16days.39yrs.min{1}'.format(index, ext)
    print(newfilename_16days_std)

    if arcpy.Exists(os.path.join(lta_folder, newfilename_16days_std)):
        print(newfilename_16days_std + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_std = arcpy.sa.CellStatistics(listoffile, "STD", "DATA")
        outCellStatistics_std.save(os.path.join(lta_folder, newfilename_16days_std))
        arcpy.CheckInExtension("spatial")

    if arcpy.Exists(os.path.join(lta_folder, newfilename_16days_avg)):
        print(newfilename_16days_avg + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_avg = arcpy.sa.CellStatistics(listoffile, "MEAN", "DATA")
        outCellStatistics_avg.save(os.path.join(lta_folder, newfilename_16days_avg))
        arcpy.CheckInExtension("spatial")

    if arcpy.Exists(os.path.join(lta_folder, newfilename_16days_max)):
        print(newfilename_16days_max + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_max = arcpy.sa.CellStatistics(listoffile, "MAXIMUM", "DATA")
        outCellStatistics_max.save(os.path.join(lta_folder, newfilename_16days_max))
        arcpy.CheckInExtension("spatial")

    if arcpy.Exists(os.path.join(lta_folder, newfilename_16days_min)):
        print(newfilename_16days_min + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_min = arcpy.sa.CellStatistics(listoffile, "MINIMUM", "DATA")
        outCellStatistics_min.save(os.path.join(lta_folder, newfilename_16days_min))
        arcpy.CheckInExtension("spatial")