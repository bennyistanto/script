import os
import arcpy
from arcpy.sa import *
from datetime import date, timedelta

# to execute first DSLR data
def execute_first_DSLR(_tiffolder, _DSLRFolder, threshold):
    sr = arcpy.SpatialReference(4326)
    print("looking at the first daily rainfall data in tif folder...")
    daily_list = create_daily_List(_tiffolder)
    first_date = min(daily_list)
    print("execute first rainy data from date "+first_date)
    first_data_name = 'chirps-v2.0.{0}.{1}.{2}.tif'.format(first_date[0:4], first_date[4:6], first_date[6:8])
    first_daily_data = os.path.join(_tiffolder, first_data_name)
    daily_Date = date(int(first_date[0:4]), int(first_date[4:6]), int(first_date[6:8]))
    dslr_date = daily_Date + timedelta(days=1)
    print("creating dslr data "+str(dslr_date)+ " from daily rainfall data from "+str(daily_Date))
    DSLRYear = str(dslr_date.year)
    DSLRmonth = str(dslr_date.month)
    DSLRday = str(dslr_date.day)
    print(str(dslr_date))
    DSLRFilename = 'cli_chirps_dslr_{0}{1}{2}.tif'.format(DSLRYear.zfill(4), DSLRmonth.zfill(2), DSLRday.zfill(2))
    print("Processing "+DSLRFilename)
    arcpy.CheckOutExtension("spatial")
    outCon = Con(Raster(first_daily_data) < int(threshold), 1, 0)
    outCon.save(os.path.join(_DSLRFolder, DSLRFilename))
    arcpy.DefineProjection_management(os.path.join(_DSLRFolder, DSLRFilename), sr)
    arcpy.CheckInExtension("spatial")
    print("file " + DSLRFilename + " is created")


# to execute next DSLR data
def execute_DSLR(_lastdate, _tiffolder, _DSLR_folder, threshold):
    sr = arcpy.SpatialReference(4326)
    date_formatted = date(int(_lastdate[0:4]), int(_lastdate[4:6]), int(_lastdate[6:8]))
    last_dslrname = 'cli_chirps_dslr_{0}'.format(_lastdate)
    last_dslrfile = os.path.join(_DSLR_folder, last_dslrname)
    next_dailyname = 'chirps-v2.0.{0}.{1}.{2}.tif'.format(_lastdate[0:4], _lastdate[4:6], _lastdate[6:8])
    next_dailydata = os.path.join(_tiffolder, next_dailyname)
    if arcpy.Exists(next_dailydata):
        print("next daily data is available...")
        print("start processing next DSLR...")
        new_dslr_date = date_formatted + timedelta(days=1)
        DSLRYear1 = str(new_dslr_date.year)
        DSLRmonth1 = str(new_dslr_date.month)
        DSLRday1 = str(new_dslr_date.day)
        new_dslr_name = 'cli_chirps_dslr_{0}{1}{2}.tif'.format(DSLRYear1.zfill(4), DSLRmonth1.zfill(2), DSLRday1.zfill(2))
        print("Processing DSLR from "+last_dslrfile+" and "+next_dailydata)
        arcpy.CheckOutExtension("spatial")
        outDSLRCon = Con(Raster(next_dailydata) < int(threshold), Raster(last_dslrfile)+1, 0)
        outDSLRCon.save(os.path.join(_DSLR_folder, new_dslr_name))
        arcpy.DefineProjection_management(os.path.join(_DSLR_folder, new_dslr_name), sr)
        arcpy.CheckInExtension("spatial")
        print("DSLR File "+new_dslr_name+" is created")
    else:
        print("next daily data is not available. Exit...")

# to check if there is DSLR data in output folder
def create_DSLR_List(_DSLR_folder):
    print("start reading existing DSLR Dataset....")
    print("looking for file with naming cli_chirps_dslr_YYYYMMDD")
    DSLR_Date_List=[]
    for DSLR_Data in os.listdir(_DSLR_folder):
        if DSLR_Data.endswith(".tif") or DSLR_Data.endswith(".tiff"):
            print("found " + DSLR_Data + " in the DSLR folder")
            parse_String = DSLR_Data.split('_')
            DSLR_Data_Date = parse_String[3]
            DSLR_Date_List.append(DSLR_Data_Date)
    return DSLR_Date_List

# to check input data
def create_daily_List(_tif_folder):
    print("start reading list of daily rainfall data....")
    print("looking for file with naming chirps-v2.0.YYYY.MM.DD")
    Daily_Date_List=[]
    for Daily_Data in os.listdir(_tif_folder):
        if Daily_Data.endswith(".tif") or Daily_Data.endswith(".tiff"):
            print("found " + Daily_Data+ " in the daily rainfall folder")
            parse_String = Daily_Data.split('.')
            Daily_Data_Date = parse_String[2]+parse_String[3]+parse_String[4]
            Daily_Date_List.append(Daily_Data_Date)
    return Daily_Date_List

# to run the script
def create_DSLR(_DSLR_folder, _tiffolder, threshold):

    DSLR_Date_List = create_DSLR_List(_DSLR_folder)
    Daily_list = create_daily_List(_tiffolder)
    # if there is no DSLR data, creating new DSLR data
    if len(DSLR_Date_List)==0:
        print("No DSLR Data found...")
        print("creating first DSLR data...")
        execute_first_DSLR(_tiffolder, _DSLR_folder, threshold)
        DSLR_Date_List = create_DSLR_List(_DSLR_folder)
    # if there is DSLR data
    print("DSLR Data found. Looking for latest DSLR Data...")
    #Check last DSLR available
    last_date = max(DSLR_Date_List)

    #Check last daily data availabke
    max_daily_date = max(Daily_list)
    last_DSLR_date = date(int(last_date[0:4]), int(last_date[4:6]), int(last_date[6:8]))
    last_daily_date = date(int(max_daily_date[0:4]), int(max_daily_date[4:6]), int(max_daily_date[6:8]))
    # process DSLR to every daily data available after last DSLR data
    while last_daily_date + timedelta(days=1) > last_DSLR_date:

        execute_DSLR(last_date, _tiffolder, _DSLR_folder, threshold)

        last_DSLR_date=last_DSLR_date+timedelta(days=1)
        DSLRYear2 = str(last_DSLR_date.year)
        DSLRmonth2 = str(last_DSLR_date.month)
        DSLRday2 = str(last_DSLR_date.day)
        last_date='{0}{1}{2}.tif'.format(DSLRYear2.zfill(4), DSLRmonth2.zfill(2), DSLRday2.zfill(2))

    print("All DSLR data is available")

# run the function (output folder, input folder, threshold)
list = create_DSLR('Z:\\Temp\\CHIRPS\\DSLR_Test\\CDD_10mm_temp','Z:\\Temp\\CHIRPS\\Daily\\2020', 10)