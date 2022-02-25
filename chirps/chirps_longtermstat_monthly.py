# -*- coding: utf-8 -*-
import os
import arcpy

# Monthly calendar
chirps_monthly_data = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

# Change the data and output folder
data_folder = "X:\\Temp\\CHIRPS\\Monthly\\geotiff"
lta_folder = "X:\\Temp\\CHIRPS\\Monthly\\lta"

dictionary = {}
for i in chirps_monthly_data:
    content = []
    for file_monthly in os.listdir(data_folder):
        if file_monthly.endswith(".tif") or file_monthly.endswith(".tiff"):
            parse_string = file_monthly.split('.')
            Dmonthseq = parse_string[3]
            if Dmonthseq == i:
                content.append(os.path.join(data_folder, file_monthly))
    dictionary[i] = content


for index in dictionary:
    listoffile = dictionary[index]
    print(listoffile)
    ext = ".tif"
    newfilename_monthly_std = 'chirps-v2.0.1981-2021.{0}.monthly.41yrs.std{1}'.format(index, ext)
    newfilename_monthly_avg = 'chirps-v2.0.1981-2021.{0}.monthly.41yrs.avg{1}'.format(index, ext)
    newfilename_monthly_max = 'chirps-v2.0.1981-2021.{0}.monthly.41yrs.max{1}'.format(index, ext)
    newfilename_monthly_min = 'chirps-v2.0.1981-2021.{0}.monthly.41yrs.min{1}'.format(index, ext)
    print(newfilename_monthly_std)

    if arcpy.Exists(os.path.join(lta_folder, newfilename_monthly_std)):
        print(newfilename_monthly_std + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_std = arcpy.sa.CellStatistics(listoffile, "STD", "DATA")
        outCellStatistics_std.save(os.path.join(lta_folder, newfilename_monthly_std))
        arcpy.CheckInExtension("spatial")

    if arcpy.Exists(os.path.join(lta_folder, newfilename_monthly_avg)):
        print(newfilename_monthly_avg + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_avg = arcpy.sa.CellStatistics(listoffile, "MEAN", "DATA")
        outCellStatistics_avg.save(os.path.join(lta_folder, newfilename_monthly_avg))
        arcpy.CheckInExtension("spatial")

    if arcpy.Exists(os.path.join(lta_folder, newfilename_monthly_max)):
        print(newfilename_monthly_max + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_max = arcpy.sa.CellStatistics(listoffile, "MAXIMUM", "DATA")
        outCellStatistics_max.save(os.path.join(lta_folder, newfilename_monthly_max))
        arcpy.CheckInExtension("spatial")

    if arcpy.Exists(os.path.join(lta_folder, newfilename_monthly_min)):
        print(newfilename_monthly_min + " exists")
    else:
        arcpy.CheckOutExtension("spatial")
        outCellStatistics_min = arcpy.sa.CellStatistics(listoffile, "MINIMUM", "DATA")
        outCellStatistics_min.save(os.path.join(lta_folder, newfilename_monthly_min))
        arcpy.CheckInExtension("spatial")