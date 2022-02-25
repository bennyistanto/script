import os
import arcpy
from arcpy.sa import *
from datetime import date, timedelta

# to execute first DSLD data
def execute_first_DSLD(_tiffolder, _DSLDFolder, threshold):
    sr = arcpy.SpatialReference(4326)
    print("looking at the first daily rainfall data in tif folder...")
    daily_list = create_daily_List(_tiffolder)
    first_date = min(daily_list)
    print("execute first rainy data from date "+first_date)
    first_data_name = 'chirps-v2.0.{0}.{1}.{2}.tif'.format(first_date[0:4], first_date[4:6], first_date[6:8])
    first_daily_data = os.path.join(_tiffolder, first_data_name)
    daily_Date = date(int(first_date[0:4]), int(first_date[4:6]), int(first_date[6:8]))
    dsld_date = daily_Date + timedelta(days=1)
    print("creating dsld data "+str(dsld_date)+ " from daily rainfall data from "+str(daily_Date))
    DSLDYear = str(dsld_date.year)
    DSLDmonth = str(dsld_date.month)
    DSLDday = str(dsld_date.day)
    print(str(dsld_date))
    DSLDFilename = 'cli_chirps_dsld_{0}{1}{2}.tif'.format(DSLDYear.zfill(4), DSLDmonth.zfill(2), DSLDday.zfill(2))
    print("Processing "+DSLDFilename)
    arcpy.CheckOutExtension("spatial")
    outCon = Con(Raster(first_daily_data) > int(threshold), 1, 0)
    outCon.save(os.path.join(_DSLDFolder, DSLDFilename))
    arcpy.DefineProjection_management(os.path.join(_DSLDFolder, DSLDFilename), sr)
    arcpy.CheckInExtension("spatial")
    print("file " + DSLDFilename + " is created")


# to execute next DSLD data
def execute_DSLD(_lastdate, _tiffolder, _DSLD_folder, threshold):
    sr = arcpy.SpatialReference(4326)
    date_formatted = date(int(_lastdate[0:4]), int(_lastdate[4:6]), int(_lastdate[6:8]))
    last_dsldname = 'cli_chirps_dsld_{0}'.format(_lastdate)
    last_dsldfile = os.path.join(_DSLD_folder, last_dsldname)
    next_dailyname = 'chirps-v2.0.{0}.{1}.{2}.tif'.format(_lastdate[0:4], _lastdate[4:6], _lastdate[6:8])
    next_dailydata = os.path.join(_tiffolder, next_dailyname)
    if arcpy.Exists(next_dailydata):
        print("next daily data is available...")
        print("start processing next DSLD...")
        new_dsld_date = date_formatted + timedelta(days=1)
        DSLDYear1 = str(new_dsld_date.year)
        DSLDmonth1 = str(new_dsld_date.month)
        DSLDday1 = str(new_dsld_date.day)
        new_dsld_name = 'cli_chirps_dsld_{0}{1}{2}.tif'.format(DSLDYear1.zfill(4), DSLDmonth1.zfill(2), DSLDday1.zfill(2))
        print("Processing DSLD from "+last_dsldfile+" and "+next_dailydata)
        arcpy.CheckOutExtension("spatial")
        outDSLDCon = Con(Raster(next_dailydata) > int(threshold), Raster(last_dsldfile)+1, 0)
        outDSLDCon.save(os.path.join(_DSLD_folder, new_dsld_name))
        arcpy.DefineProjection_management(os.path.join(_DSLD_folder, new_dsld_name), sr)
        arcpy.CheckInExtension("spatial")
        print("DSLD File "+new_dsld_name+" is created")
    else:
        print("next daily data is not available. Exit...")

# to check if there is DSLD data in output folder
def create_DSLD_List(_DSLD_folder):
    print("start reading existing DSLD Dataset....")
    print("looking for file with naming cli_chirps_dsld_YYYYMMDD")
    DSLD_Date_List=[]
    for DSLD_Data in os.listdir(_DSLD_folder):
        if DSLD_Data.endswith(".tif") or DSLD_Data.endswith(".tiff"):
            print("found " + DSLD_Data + " in the DSLD folder")
            parse_String = DSLD_Data.split('_')
            DSLD_Data_Date = parse_String[3]
            DSLD_Date_List.append(DSLD_Data_Date)
    return DSLD_Date_List

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
def create_DSLD(_DSLD_folder, _tiffolder, threshold):

    DSLD_Date_List = create_DSLD_List(_DSLD_folder)
    Daily_list = create_daily_List(_tiffolder)
    # if there is no DSLD data, creating new DSLD data
    if len(DSLD_Date_List)==0:
        print("No DSLD Data found...")
        print("creating first DSLD data...")
        execute_first_DSLD(_tiffolder, _DSLD_folder, threshold)
        DSLD_Date_List = create_DSLD_List(_DSLD_folder)
    # if there is DSLD data
    print("DSLD Data found. Looking for latest DSLD Data...")
    #Check last DSLD available
    last_date = max(DSLD_Date_List)
    print("last date"+last_date)
    #Check last daily data availabke
    max_daily_date = max(Daily_list)
    last_DSLD_date = date(int(last_date[0:4]), int(last_date[4:6]), int(last_date[6:8]))
    last_daily_date = date(int(max_daily_date[0:4]), int(max_daily_date[4:6]), int(max_daily_date[6:8]))
    # process DSLD to every daily data available after last DSLD data
    while last_daily_date + timedelta(days=1) > last_DSLD_date:
        print("Latest DSLD is from "+last_date)
        execute_DSLD(last_date, _tiffolder, _DSLD_folder, threshold)

        last_DSLD_date=last_DSLD_date+timedelta(days=1)
        DSLDYear2 = str(last_DSLD_date.year)
        DSLDmonth2 = str(last_DSLD_date.month)
        DSLDday2 = str(last_DSLD_date.day)
        last_date='{0}{1}{2}.tif'.format(DSLDYear2.zfill(4), DSLDmonth2.zfill(2), DSLDday2.zfill(2))

    print("All DSLD data is available")

# run the function (output folder, input folder, threshold)
list = create_DSLD('Z:\\Temp\\CHIRPS\\DSLR_Test\\CWD_5mm_temp','Z:\\Temp\\CHIRPS\\Daily\\2020', 5)