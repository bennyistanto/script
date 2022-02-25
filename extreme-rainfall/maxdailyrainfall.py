# -*- coding: utf-8 -*-
import arcpy
import os
from datetime import date
from datetime import timedelta
import multiprocessing

#Overwrite the output if exist
arcpy.env.overwriteOutput = True

#Environment variable
input_folder = 'Z:\\Temp\\CHIRPS\\Daily\\Data\\1994' #Adjust the input directory
output_folder = 'Z:\\Temp\\CHIRPS\\Daily\\Data' #Adjust the output directory
year = '1994' #Adjust year

#Extract max rainfall
def extract_max(folder_to_extract, output_folder, year):
    listoffile = []
    for data in os.listdir(folder_to_extract):
        if data.endswith(".tif"):
            listoffile.append(os.path.join(folder_to_extract, data))
    print("data to calculate is "+str(len(listoffile)))
    print("start running cell statistics to find maximum rainfall rate in "+year+" .....")
    arcpy.CheckOutExtension("spatial")
    max_data_filename = 'chirps-v.2.0.{0}.1days.max.tif'.format(year)
    max_data = arcpy.sa.CellStatistics(listoffile, "MAXIMUM", "DATA")
    max_data.save(os.path.join(output_folder, max_data_filename))
    print(max_data_filename + ' is succesfully created')
    arcpy.CheckInExtension("spatial")

if __name__ == '__main__':
    print("input folder "+input_folder)
    print("output folder "+output_folder)
    extract_max(input_folder, output_folder, year)