# -*- coding: utf-8 -*-
"""
NAME
    imerg_daily_indices_1mm.py
    Global IMERG daily indices data extraction
DESCRIPTION
    Input data for this script will use IMERG Final or Late Run downloaded from NASA website
    This script can do:
        (i)    Dry days
        (ii)   Wet days
        (iii)  Consecutive dry days
        (iv)   Consecutive wet days
PROCESS
    (i)   Extract IMERG's daily rainfall with value greater/less than 1mm (threshold for a day categoried as rainy day)
    (ii)  If Rainfall < 1 = Yes, then assign 1 otherwise 0. This is for dry condition, for wet are the opposite.
    (iii) For number of consecutive information, it will accumulate to next data calculation result,
          if the value = 1. If not, start from 0 again.
REQUIREMENT
    ArcGIS must installed before using this script, as it required arcpy module.
EXAMPLES
    C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python imerg_daily_indices.py
NOTES
    This script is designed to work with global IMERG data (Final or Late Run), following this guideline
    https://github.com/bennyistanto/climate-data-application/blob/main/CDD_using_IMERG_data.md
    If using other data, some adjustment are required: parsing filename, directory, threshold
    All IMERG data in nc4 format are available at JNB Server Drive J:\\Data\\GLOBAL\\CLIMATE\\imerg\\nc4\\
CONTACT
    Benny Istanto
    Climate Geographer
    GOST, The World Bank
LICENSE
    This script is in the public domain, free from copyrights or restrictions.
VERSION
    $Id$
TODO
    xx
"""
import arcpy
from arcpy.sa import *
import os, sys, traceback
import re
from datetime import date, timedelta


# To avoid overwriting outputs, change overwriteOutput option to False.
arcpy.env.overwriteOutput = True

# Check precipitation data in GeoTIFF format
def create_daily_List(_tif_folder):
    print("start reading list of daily Rainfall data....")
    print("looking for file with naming wld_cli_1d_imerg_YYYYMMDD.tif")
    
    Daily_Date_List=[]
    
    for Daily_Data in os.listdir(_tif_folder):
        
        if Daily_Data.endswith(".tif"):
            print("found " + Daily_Data+ " in the daily Rainfall folder")
            
            i_imerg = Daily_Data.index('imerg_')
            ymd = Daily_Data[i_imerg + 6:i_imerg+6+8]
            
            Daily_Data_Date = ymd            
            Daily_Date_List.append(Daily_Data_Date)
    
    return sorted(Daily_Date_List)


#############
# DRY DAYS
# Check if there is DRY Days data in output folder
def create_DD_List(_DD_folder):
    print("start reading existing DRY Days Dataset....")
    print("looking for file with naming wld_cli_dry_1mm_imerg_YYYYMMDD")
    
    DD_Date_List=[]
    
    for DD_Data in os.listdir(_DD_folder):
        if DD_Data.endswith(".tif"):
            print("found " + DD_Data + " in the DRY Days folder")
            
            parse_String = DD_Data.split('_')
            DD_Data_Date = parse_String[5]
            DD_Date_List.append(DD_Data_Date)
    
    return DD_Date_List


# Execute first Dry condition
def execute_first_DD(_tiffolder, _DDFolder, threshold):
    # Spatial reference WGS-84
    sr = arcpy.SpatialReference(4326)
    print("looking at the first daily Rainfall data in tif folder...")
    
    daily_list = create_daily_List(_tiffolder)
    first_date = min(daily_list)
    print("execute first Rainfall data from date "+first_date)
    
    first_data_name = 'wld_cli_1d_imerg_{0}{1}{2}.tif'.format(first_date[0:4], first_date[4:6], first_date[6:8])
    first_daily_data = os.path.join(_tiffolder, first_data_name)
    daily_Date = date(int(first_date[0:4]), int(first_date[4:6]), int(first_date[6:8]))
    dry_date = daily_Date #+ timedelta(days=1)
    print("creating dry data "+str(dry_date)+ " from daily Rainfall data from "+str(daily_Date))
    
    DDyear = str(dry_date.year)
    DDmonth = str(dry_date.month)
    DDday = str(dry_date.day)
    print(str(dry_date))
    
    DDFilename = 'wld_cli_dry_1mm_imerg_{0}{1}{2}.tif'.format(DDyear.zfill(4), DDmonth.zfill(2), DDday.zfill(2))
    print("Processing "+DDFilename)
    
    arcpy.CheckOutExtension("spatial")
    
    outCon = Con(Raster(first_daily_data) < Float(threshold), Float(1), Float(0))
    outCon.save(os.path.join(_DDFolder, DDFilename))
    arcpy.DefineProjection_management(os.path.join(_DDFolder, DDFilename), sr)
    
    arcpy.CheckInExtension("spatial")
    print("file " + DDFilename + " is created")


# Execute next DRY Days data
def execute_DD(_lastdate, _tiffolder, _DD_folder, threshold):
    # Spatial reference WGS-84
    sr = arcpy.SpatialReference(4326)
    
    date_formatted = date(int(_lastdate[0:4]), int(_lastdate[4:6]), int(_lastdate[6:8]))
    last_dryname = 'wld_cli_dry_1mm_imerg_{0}'.format(_lastdate)
    last_dryfile = os.path.join(_DD_folder, last_dryname)
    next_daily_date = date_formatted + timedelta(days=1)
    next_dailyname = 'wld_cli_1d_imerg_{0}.tif'.format(next_daily_date.strftime('%Y%m%d'))
    #next_dailyname = 'wld_cli_1d_imerg_{0}{1}{2}.tif'.format(_lastdate[0:4], _lastdate[4:6], _lastdate[6:8])
    next_dailydata = os.path.join(_tiffolder, next_dailyname)
    
    if arcpy.Exists(next_dailydata):
        print("next daily data is available...")
        print("start processing next DRY Days...")
        
        new_dry_date = date_formatted + timedelta(days=1)
        DDyear1 = str(new_dry_date.year)
        DDmonth1 = str(new_dry_date.month)
        DDday1 = str(new_dry_date.day)
        new_dry_name = 'wld_cli_dry_1mm_imerg_{0}{1}{2}.tif'.format(DDyear1.zfill(4), DDmonth1.zfill(2), DDday1.zfill(2))
        print("Processing DRY Days from "+next_dailydata)
        
        arcpy.CheckOutExtension("spatial")
        
        outDDCon = Con(Raster(next_dailydata) < Float(threshold), Float(1), Float(0))
        outDDCon.save(os.path.join(_DD_folder, new_dry_name))
        arcpy.DefineProjection_management(os.path.join(_DD_folder, new_dry_name), sr)
        
        arcpy.CheckInExtension("spatial")
        print("DRY Days File "+new_dry_name+" is created")
    
    else:
        print("next daily data is not available. Exit...")


# Run the script
def create_DD(_DD_folder, _tiffolder, threshold):

    DD_Date_List = create_DD_List(_DD_folder)
    Daily_list = create_daily_List(_tiffolder)
    
    # if there is no DRY data, creating new DRY data
    if len(DD_Date_List)==0:
        print("No DRY Days data found...")
        print("Creating first DRY Days data...")
        
        execute_first_DD(_tiffolder, _DD_folder, threshold)
        DD_Date_List = create_DD_List(_DD_folder)
    
    # if there is DRY Days data
    print("DRY Days data found. Looking for latest DRY Days data...")
    
    #Check last DRY Days available
    last_date = max(DD_Date_List)

    #Check last daily data availabke
    max_daily_date = max(Daily_list)
    last_DD_date = date(int(last_date[0:4]), int(last_date[4:6]), int(last_date[6:8]))
    last_daily_date = date(int(max_daily_date[0:4]), int(max_daily_date[4:6]), int(max_daily_date[6:8]))
    
    # process DRY Days to every daily data available after last DRY Days data
    while last_daily_date + timedelta(days=1) > last_DD_date:
    #while last_daily_date > last_DD_date:

        execute_DD(last_date, _tiffolder, _DD_folder, threshold)

        last_DD_date=last_DD_date+timedelta(days=1)
        DDyear2 = str(last_DD_date.year)
        DDmonth2 = str(last_DD_date.month)
        DDday2 = str(last_DD_date.day)
        last_date='{0}{1}{2}.tif'.format(DDyear2.zfill(4), DDmonth2.zfill(2), DDday2.zfill(2))

    print("All DRY Days data is available")


#############
# WET DAYS
# Check if there is WET Days data in output folder
def create_WD_List(_WD_folder):
    print("start reading existing WET Days Dataset....")
    print("looking for file with naming wld_cli_wet_1mm_imerg_YYYYMMDD")
    
    WD_Date_List=[]
    
    for WD_Data in os.listdir(_WD_folder):
        if WD_Data.endswith(".tif"):
            print("found " + WD_Data + " in the WET Days folder")
            
            parse_String = WD_Data.split('_')
            WD_Data_Date = parse_String[5]
            WD_Date_List.append(WD_Data_Date)
    
    return WD_Date_List


# Execute first WET condition
def execute_first_WD(_tiffolder, _WDFolder, threshold):
    # Spatial reference WGS-84
    sr = arcpy.SpatialReference(4326)
    print("looking at the first daily Rainfall data in tif folder...")
    
    daily_list = create_daily_List(_tiffolder)
    first_date = min(daily_list)
    print("execute first Rainfall data from date "+first_date)
    
    first_data_name = 'wld_cli_1d_imerg_{0}{1}{2}.tif'.format(first_date[0:4], first_date[4:6], first_date[6:8])
    first_daily_data = os.path.join(_tiffolder, first_data_name)
    daily_Date = date(int(first_date[0:4]), int(first_date[4:6]), int(first_date[6:8]))
    wet_date = daily_Date #+ timedelta(days=1)
    print("creating wet data "+str(wet_date)+ " from daily Rainfall data from "+str(daily_Date))
    
    WDyear = str(wet_date.year)
    WDmonth = str(wet_date.month)
    WDday = str(wet_date.day)
    print(str(wet_date))
    
    WDFilename = 'wld_cli_wet_1mm_imerg_{0}{1}{2}.tif'.format(WDyear.zfill(4), WDmonth.zfill(2), WDday.zfill(2))
    print("Processing "+WDFilename)
    
    arcpy.CheckOutExtension("spatial")
    
    outCon = Con(Raster(first_daily_data) < Float(threshold), Float(0), Float(1))
    outCon.save(os.path.join(_WDFolder, WDFilename))
    arcpy.DefineProjection_management(os.path.join(_WDFolder, WDFilename), sr)
    
    arcpy.CheckInExtension("spatial")
    print("file " + WDFilename + " is created")


# Execute next WET Days data
def execute_WD(_lastdate, _tiffolder, _WD_folder, threshold):
    # Spatial reference WGS-84
    sr = arcpy.SpatialReference(4326)
    
    date_formatted = date(int(_lastdate[0:4]), int(_lastdate[4:6]), int(_lastdate[6:8]))
    last_wetname = 'wld_cli_wet_1mm_imerg_{0}'.format(_lastdate)
    last_wetfile = os.path.join(_WD_folder, last_wetname)
    next_daily_date = date_formatted + timedelta(days=1)
    next_dailyname = 'wld_cli_1d_imerg_{0}.tif'.format(next_daily_date.strftime('%Y%m%d'))
    #next_dailyname = 'wld_cli_1d_imerg_{0}{1}{2}.tif'.format(_lastdate[0:4], _lastdate[4:6], _lastdate[6:8])
    next_dailydata = os.path.join(_tiffolder, next_dailyname)
    
    if arcpy.Exists(next_dailydata):
        print("next daily data is available...")
        print("start processing next WET Days...")
        
        new_wet_date = date_formatted + timedelta(days=1)
        WDyear1 = str(new_wet_date.year)
        WDmonth1 = str(new_wet_date.month)
        WDday1 = str(new_wet_date.day)
        new_wet_name = 'wld_cli_wet_1mm_imerg_{0}{1}{2}.tif'.format(WDyear1.zfill(4), WDmonth1.zfill(2), WDday1.zfill(2))
        print("Processing WET Days from "+next_dailydata)
        
        arcpy.CheckOutExtension("spatial")
        
        outWDCon = Con(Raster(next_dailydata) < Float(threshold), Float(0), Float(1))
        outWDCon.save(os.path.join(_WD_folder, new_wet_name))
        arcpy.DefineProjection_management(os.path.join(_WD_folder, new_wet_name), sr)
        
        arcpy.CheckInExtension("spatial")
        print("WET Days File "+new_wet_name+" is created")
    
    else:
        print("next daily data is not available. Exit...")


# Run the script
def create_WD(_WD_folder, _tiffolder, threshold):

    WD_Date_List = create_WD_List(_WD_folder)
    Daily_list = create_daily_List(_tiffolder)
    
    # if there is no WET data, creating new WET data
    if len(WD_Date_List)==0:
        print("No WET Days data found...")
        print("Creating first WET Days data...")
        
        execute_first_WD(_tiffolder, _WD_folder, threshold)
        WD_Date_List = create_WD_List(_WD_folder)
    
    # if there is WET Days data
    print("WET Days data found. Looking for latest WET Days data...")
    
    #Check last WET Days available
    last_date = max(WD_Date_List)

    #Check last daily data availabke
    max_daily_date = max(Daily_list)
    last_WD_date = date(int(last_date[0:4]), int(last_date[4:6]), int(last_date[6:8]))
    last_daily_date = date(int(max_daily_date[0:4]), int(max_daily_date[4:6]), int(max_daily_date[6:8]))
    
    # process WET Days to every daily data available after last WET Days data
    while last_daily_date + timedelta(days=1) > last_WD_date:
    #while last_daily_date > last_WD_date:    

        execute_WD(last_date, _tiffolder, _WD_folder, threshold)

        last_WD_date=last_WD_date+timedelta(days=1)
        WDyear2 = str(last_WD_date.year)
        WDmonth2 = str(last_WD_date.month)
        WDday2 = str(last_WD_date.day)
        last_date='{0}{1}{2}.tif'.format(WDyear2.zfill(4), WDmonth2.zfill(2), WDday2.zfill(2))

    print("All WET Days data is available")


#############
# CONSECUTIVE DRY DAYS
# Check if there is Consecutive DRY Days data in output folder
def create_CDD_List(_CDD_folder):
    print("start reading existing Consecutive DRY Days Dataset....")
    print("looking for file with naming wld_cli_cdd_1mm_imerg_YYYYMMDD")
    
    CDD_Date_List=[]
    
    for CDD_Data in os.listdir(_CDD_folder):
        if CDD_Data.endswith(".tif"):
            print("found " + CDD_Data + " in the Consecutive DRY Days folder")
            
            parse_String = CDD_Data.split('_')
            CDD_Data_Date = parse_String[5]
            CDD_Date_List.append(CDD_Data_Date)
    
    return CDD_Date_List


# Execute first Dry condition
def execute_first_CDD(_tiffolder, _CDDFolder, threshold):
    # Spatial reference WGS-84
    sr = arcpy.SpatialReference(4326)
    print("looking at the first daily Rainfall data in tif folder...")
    
    daily_list = create_daily_List(_tiffolder)
    first_date = min(daily_list)
    print("execute first Rainfall data from date "+first_date)
    
    first_data_name = 'wld_cli_1d_imerg_{0}{1}{2}.tif'.format(first_date[0:4], first_date[4:6], first_date[6:8])
    first_daily_data = os.path.join(_tiffolder, first_data_name)
    daily_Date = date(int(first_date[0:4]), int(first_date[4:6]), int(first_date[6:8]))
    dry_date = daily_Date #+ timedelta(days=1)
    print("creating dry data "+str(dry_date)+ " from daily Rainfall data from "+str(daily_Date))
    
    CDDyear = str(dry_date.year)
    CDDmonth = str(dry_date.month)
    CDDday = str(dry_date.day)
    print(str(dry_date))
    
    CDDFilename = 'wld_cli_cdd_1mm_imerg_{0}{1}{2}.tif'.format(CDDyear.zfill(4), CDDmonth.zfill(2), CDDday.zfill(2))
    print("Processing "+CDDFilename)
    
    arcpy.CheckOutExtension("spatial")
    
    outCon = Con(Raster(first_daily_data) < Float(threshold), Float(1), Float(0))
    outCon.save(os.path.join(_CDDFolder, CDDFilename))
    arcpy.DefineProjection_management(os.path.join(_CDDFolder, CDDFilename), sr)
    
    arcpy.CheckInExtension("spatial")
    print("file " + CDDFilename + " is created")


# Execute next Consecutive DRY Days data
def execute_CDD(_lastdate, _tiffolder, _CDD_folder, threshold):
    # Spatial reference WGS-84
    sr = arcpy.SpatialReference(4326)
    
    date_formatted = date(int(_lastdate[0:4]), int(_lastdate[4:6]), int(_lastdate[6:8]))
    last_dryname = 'wld_cli_cdd_1mm_imerg_{0}'.format(_lastdate)
    last_dryfile = os.path.join(_CDD_folder, last_dryname)
    next_daily_date = date_formatted + timedelta(days=1)
    next_dailyname = 'wld_cli_1d_imerg_{0}.tif'.format(next_daily_date.strftime('%Y%m%d'))
    #next_dailyname = 'wld_cli_1d_imerg_{0}{1}{2}.tif'.format(_lastdate[0:4], _lastdate[4:6], _lastdate[6:8])
    next_dailydata = os.path.join(_tiffolder, next_dailyname)
    
    if arcpy.Exists(next_dailydata):
        print("next daily data is available...")
        print("start processing next Consecutive DRY Days...")
        
        new_dry_date = date_formatted + timedelta(days=1)
        CDDyear1 = str(new_dry_date.year)
        CDDmonth1 = str(new_dry_date.month)
        CDDday1 = str(new_dry_date.day)
        new_dry_name = 'wld_cli_cdd_1mm_imerg_{0}{1}{2}.tif'.format(CDDyear1.zfill(4), CDDmonth1.zfill(2), CDDday1.zfill(2))
        print("Processing Consecutive DRY Days from "+last_dryfile+" and "+next_dailydata)
        
        arcpy.CheckOutExtension("spatial")
        
        outCDDCon = Con(Raster(next_dailydata) < Float(threshold), Raster(last_dryfile)+Float(1), Float(0))
        outCDDCon.save(os.path.join(_CDD_folder, new_dry_name))
        arcpy.DefineProjection_management(os.path.join(_CDD_folder, new_dry_name), sr)
        
        arcpy.CheckInExtension("spatial")
        print("Consecutive DRY Days File "+new_dry_name+" is created")
    
    else:
        print("next daily data is not available. Exit...")


# Run the script
def create_CDD(_CDD_folder, _tiffolder, threshold):

    CDD_Date_List = create_CDD_List(_CDD_folder)
    Daily_list = create_daily_List(_tiffolder)
    
    # if there is no DRY data, creating new DRY data
    if len(CDD_Date_List)==0:
        print("No Consecutive DRY Days data found...")
        print("Creating first Consecutive DRY Days data...")
        
        execute_first_CDD(_tiffolder, _CDD_folder, threshold)
        CDD_Date_List = create_CDD_List(_CDD_folder)
    
    # if there is Consecutive DRY Days data
    print("Consecutive DRY Days data found. Looking for latest Consecutive DRY Days data...")
    
    #Check last Consecutive DRY Days available
    last_date = max(CDD_Date_List)

    #Check last daily data availabke
    max_daily_date = max(Daily_list)
    last_CDD_date = date(int(last_date[0:4]), int(last_date[4:6]), int(last_date[6:8]))
    last_daily_date = date(int(max_daily_date[0:4]), int(max_daily_date[4:6]), int(max_daily_date[6:8]))
    
    # process Consecutive DRY Days to every daily data available after last Consecutive DRY Days data
    while last_daily_date + timedelta(days=1) > last_CDD_date:
    #while last_daily_date > last_CDD_date:

        execute_CDD(last_date, _tiffolder, _CDD_folder, threshold)

        last_CDD_date=last_CDD_date+timedelta(days=1)
        CDDyear2 = str(last_CDD_date.year)
        CDDmonth2 = str(last_CDD_date.month)
        CDDday2 = str(last_CDD_date.day)
        last_date='{0}{1}{2}.tif'.format(CDDyear2.zfill(4), CDDmonth2.zfill(2), CDDday2.zfill(2))

    print("All Consecutive DRY Days data is available")


#############
# CONSECUTIVE WET DAYS
# Check if there is Consecutive WET Days data in output folder
def create_CWD_List(_CWD_folder):
    print("start reading existing Consecutive WET Days Dataset....")
    print("looking for file with naming wld_cli_cwd_1mm_imerg_YYYYMMDD")
    
    CWD_Date_List=[]
    
    for CWD_Data in os.listdir(_CWD_folder):
        if CWD_Data.endswith(".tif"):
            print("found " + CWD_Data + " in the Consecutive WET Days folder")
            
            parse_String = CWD_Data.split('_')
            CWD_Data_Date = parse_String[5]
            CWD_Date_List.append(CWD_Data_Date)
    
    return CWD_Date_List


# Execute first Wet condition
def execute_first_CWD(_tiffolder, _CWDFolder, threshold):
    # Spatial reference WGS-84
    sr = arcpy.SpatialReference(4326)
    print("looking at the first daily Rainfall data in tif folder...")
    
    daily_list = create_daily_List(_tiffolder)
    first_date = min(daily_list)
    print("execute first Rainfall data from date "+first_date)
    
    first_data_name = 'wld_cli_1d_imerg_{0}{1}{2}.tif'.format(first_date[0:4], first_date[4:6], first_date[6:8])
    first_daily_data = os.path.join(_tiffolder, first_data_name)
    daily_Date = date(int(first_date[0:4]), int(first_date[4:6]), int(first_date[6:8]))
    wet_date = daily_Date + timedelta(days=1)
    print("creating wet data "+str(wet_date)+ " from daily Rainfall data from "+str(daily_Date))
    
    CWDyear = str(wet_date.year)
    CWDmonth = str(wet_date.month)
    CWDday = str(wet_date.day)
    print(str(wet_date))
    
    CWDFilename = 'wld_cli_cwd_1mm_imerg_{0}{1}{2}.tif'.format(CWDyear.zfill(4), CWDmonth.zfill(2), CWDday.zfill(2))
    print("Processing "+CWDFilename)
    
    arcpy.CheckOutExtension("spatial")
    
    outCon = Con(Raster(first_daily_data) < Float(threshold), Float(0), Float(1))
    outCon.save(os.path.join(_CWDFolder, CWDFilename))
    arcpy.DefineProjection_management(os.path.join(_CWDFolder, CWDFilename), sr)
    
    arcpy.CheckInExtension("spatial")
    print("file " + CWDFilename + " is created")


# Execute next Consecutive DRY Days data
def execute_CWD(_lastdate, _tiffolder, _CWD_folder, threshold):
    # Spatial reference WGS-84
    sr = arcpy.SpatialReference(4326)
    
    date_formatted = date(int(_lastdate[0:4]), int(_lastdate[4:6]), int(_lastdate[6:8]))
    last_wetname = 'wld_cli_cwd_1mm_imerg_{0}'.format(_lastdate)
    last_wetfile = os.path.join(_CWD_folder, last_wetname)
    next_daily_date = date_formatted + timedelta(days=1)
    next_dailyname = 'wld_cli_1d_imerg_{0}.tif'.format(next_daily_date.strftime('%Y%m%d'))
    #next_dailyname = 'wld_cli_1d_imerg_{0}{1}{2}.tif'.format(_lastdate[0:4], _lastdate[4:6], _lastdate[6:8])
    next_dailydata = os.path.join(_tiffolder, next_dailyname)
    
    if arcpy.Exists(next_dailydata):
        print("next daily data is available...")
        print("start processing next Consecutive WET Days...")
        
        new_wet_date = date_formatted + timedelta(days=1)
        CWDyear1 = str(new_wet_date.year)
        CWDmonth1 = str(new_wet_date.month)
        CWDday1 = str(new_wet_date.day)
        new_wet_name = 'wld_cli_cwd_1mm_imerg_{0}{1}{2}.tif'.format(CWDyear1.zfill(4), CWDmonth1.zfill(2), CWDday1.zfill(2))
        print("Processing Consecutive WET Days from "+last_wetfile+" and "+next_dailydata)
        
        arcpy.CheckOutExtension("spatial")
        
        outCWDCon = Con(Raster(next_dailydata) < Float(threshold), Float(0), Raster(last_wetfile)+Float(1))
        outCWDCon.save(os.path.join(_CWD_folder, new_wet_name))
        arcpy.DefineProjection_management(os.path.join(_CWD_folder, new_wet_name), sr)
        
        arcpy.CheckInExtension("spatial")
        print("Consecutive WET Days File "+new_wet_name+" is created")
    
    else:
        print("next daily data is not available. Exit...")


# Run the script
def create_CWD(_CWD_folder, _tiffolder, threshold):

    CWD_Date_List = create_CWD_List(_CWD_folder)
    Daily_list = create_daily_List(_tiffolder)
    
    # if there is no WET data, creating new WET data
    if len(CWD_Date_List)==0:
        print("No Consecutive WET Days data found...")
        print("Creating first Consecutive WET Days data...")
        
        execute_first_CWD(_tiffolder, _CWD_folder, threshold)
        CWD_Date_List = create_CWD_List(_CWD_folder)
    
    # if there is Consecutive WET Days data
    print("Consecutive WET Days data found. Looking for latest Consecutive WET Days data...")
    
    #Check last Consecutive WET Days available
    last_date = max(CWD_Date_List)

    #Check last daily data availabke
    max_daily_date = max(Daily_list)
    last_CWD_date = date(int(last_date[0:4]), int(last_date[4:6]), int(last_date[6:8]))
    last_daily_date = date(int(max_daily_date[0:4]), int(max_daily_date[4:6]), int(max_daily_date[6:8]))
    
    # process Consecutive WET Days to every daily data available after last Consecutive WET Days data
    while last_daily_date + timedelta(days=1) > last_CWD_date:
    #while last_daily_date > last_CWD_date:

        execute_CWD(last_date, _tiffolder, _CWD_folder, threshold)

        last_CWD_date=last_CWD_date+timedelta(days=1)
        CWDyear2 = str(last_CWD_date.year)
        CWDmonth2 = str(last_CWD_date.month)
        CWDday2 = str(last_CWD_date.day)
        last_date='{0}{1}{2}.tif'.format(CWDyear2.zfill(4), CWDmonth2.zfill(2), CWDday2.zfill(2))

    print("All Consecutive WET Days data is available")


# Let's go!
if __name__ == '__main__':
    # Global Environment settings
    with arcpy.EnvManager(scratchWorkspace=r"X:\ArcGIS_TEMP\Scratch.gdb", \
        workspace=r"X:\ArcGIS_TEMP\Default.gdb"):
        # Run the function (output folder, input folder, threshold)
        create_CDD('X:\\Temp\\imerg\\products\\cdd\\cdd_1mm_temp',\
            'X:\\Temp\\imerg\\data\\geotiff\\rainfall_1days', 1)
        # Run the function (output folder, input folder, threshold)
        create_DD('X:\\Temp\\imerg\\products\\drydays\\drydays_1mm_temp',\
            'X:\\Temp\\imerg\\data\\geotiff\\rainfall_1days', 1)
        # Run the function (output folder, input folder, threshold)
        create_CWD('X:\\Temp\\imerg\\products\\cwd\\cwd_1mm_temp',\
            'X:\\Temp\\imerg\\data\\geotiff\\rainfall_1days', 1)
        # Run the function (output folder, input folder, threshold)
        create_WD('X:\\Temp\\imerg\\products\\wetdays\\wetdays_1mm_temp',\
            'X:\\Temp\\imerg\\data\\geotiff\\rainfall_1days', 1)