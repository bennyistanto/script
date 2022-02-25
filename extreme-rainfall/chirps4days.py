import arcpy
import os
from datetime import date
from datetime import timedelta
import multiprocessing

#Overwrite the output if exist
arcpy.env.overwriteOutput = True

#--------------------- environment variable ---------------------#
input_folder = 'Z:\\Temp\\CHIRPS\\Daily\\Data\\2020'
output_folder = 'Z:\\Temp\\CHIRPS\\Daily\\Day4'
year = '2020'

#--------------------- function definition ----------------------#

def sum_4_days(data_folder, output_folder):
    for i in os.listdir(data_folder):
        if i.endswith(".tif"):
            parseString = i.split('.')
            data_year = parseString[2]
            data_month = parseString[3]
            data_day = parseString[4]
            data_date = date(int(data_year), int(data_month), int(data_day))
            data_2_date = data_date + timedelta(days=1)
            data_3_date = data_date + timedelta(days=2)
            data_4_date = data_date + timedelta(days=3)
            data_file_1 = os.path.join(data_folder, 'chirps-v2.0.'+str(data_date.year)+'.'
                                       +str(data_date.month).zfill(2)+'.'
                                       +str(data_date.day).zfill(2)+'.tif')
            data_file_2 = os.path.join(data_folder, 'chirps-v2.0.' + str(data_2_date.year) + '.'
                                       + str(data_2_date.month).zfill(2) + '.'
                                       + str(data_2_date.day).zfill(2) + '.tif')
            data_file_3 = os.path.join(data_folder, 'chirps-v2.0.' + str(data_3_date.year) + '.'
                                       + str(data_3_date.month).zfill(2) + '.'
                                       + str(data_3_date.day).zfill(2) + '.tif')
            data_file_4 = os.path.join(data_folder, 'chirps-v2.0.' + str(data_4_date.year) + '.'
                                       + str(data_4_date.month).zfill(2) + '.'
                                       + str(data_4_date.day).zfill(2) + '.tif')
            data_file_1234_name = 'chirps-v2.0.{0}{1}{2}.4days.tif'.format(str(data_4_date.year),
                                                                            str(data_4_date.month).zfill(2),
                                                                            str(data_4_date.day).zfill(2))
            if os.path.exists(data_file_1) and os.path.exists(data_file_2) and os.path.exists(data_file_3) and os.path.exists(data_file_4):
                print(str(data_date)+" next 3 days data exist. Start calculating....")
                arcpy.CheckOutExtension("spatial")
                with_null_1 = arcpy.sa.SetNull(data_file_1 < 0, data_file_1)
                with_null_2 = arcpy.sa.SetNull(data_file_2 < 0, data_file_2)
                with_null_3 = arcpy.sa.SetNull(data_file_3 < 0, data_file_3)
                with_null_4 = arcpy.sa.SetNull(data_file_4 < 0, data_file_4)
                data_1234 = arcpy.sa.CellStatistics([with_null_1, with_null_2, with_null_3, with_null_4], "SUM", "DATA")
                data_1234.save(os.path.join(output_folder, data_file_1234_name))
                print(data_file_1234_name+ ' is succesfully created')
                arcpy.CheckInExtension("spatial")
            else:
                print(str(data_date)+" does not have complete 3 following data")

def extract_max(folder_to_extract, output_folder, year):
    listoffile = []
    for data in os.listdir(folder_to_extract):
        if data.endswith(".tif"):
            listoffile.append(os.path.join(folder_to_extract, data))
    print("data to calculate is "+str(len(listoffile)))
    print("start running cell statistics to find maximum rainfall rate in "+year+" .....")
    arcpy.CheckOutExtension("spatial")
    max_data_filename = 'chirps-v.2.0.{0}.4days.max.tif'.format(year)
    max_data = arcpy.sa.CellStatistics(listoffile, "MAXIMUM", "DATA")
    max_data.save(os.path.join(output_folder, max_data_filename))
    print(max_data_filename + ' is succesfully created')
    arcpy.CheckInExtension("spatial")


#----------------- calculating chirps 4 days data----------------#
if __name__ == '__main__':
    print("input folder "+input_folder)
    print("output folder "+output_folder)
    folder_4days = os.path.join(output_folder, "calc_4days_"+year)
    os.mkdir(folder_4days)
    print(folder_4days+" is succesfully created")
    print("start calculating.........")
    sum_4_days(input_folder, folder_4days)
    extract_max(folder_4days, output_folder, year)