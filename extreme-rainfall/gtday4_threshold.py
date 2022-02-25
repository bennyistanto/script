#!/usr/local/bin/python

import arcpy
import os
import numpy
import pandas as pd
import glob
from arcpy.sa import *

#---------------------------Set environment settings------------------------------------#

year = '2011'

#----------------------------------Set Input Folder-------------------------------------#
returnperiod_folder = 'Z:\\Temp\\CHIRPS\\Extreme\\Threshold\\Pct4'
rainfall_folder = 'Z:\\Temp\\CHIRPS\\Extreme\\Day4\\2011'
output_output_folder = 'Z:\\Temp\\CHIRPS\\Extreme\\Day4_output\\2011_output'

#-----------------------Set Magnitude for every threshold raster------------------------#
set_magnitude = {'idn_cli_4days_precipthreshold_q0800_5yr_chirps_wfp.tif': 1,
                 'idn_cli_4days_precipthreshold_q0900_10yr_chirps_wfp.tif': 2,
                 'idn_cli_4days_precipthreshold_q0960_25yr_chirps_wfp.tif': 3,
                 'idn_cli_4days_precipthreshold_q0980_50yr_chirps_wfp.tif': 4}

#-----------------------Set folder naming for the output--------------------------------#
above_threshold_folder = os.path.join(output_output_folder, "1_rainfall_rate_above_threshold")
extremerain_magnitude_folder = output_folder = os.path.join(output_output_folder, "2_ExtremeRain_Magnitude")
max_extremerain_magnitude_folder = os.path.join(output_output_folder, "3_Max_ExtremeRain_Magnitude")
csv_version_2_folder = os.path.join(output_output_folder, '4_csv_ver_2')
csv_version_2_folder_csv = os.path.join(csv_version_2_folder, '1_csv_with_date_and_threshold')

#-------------------------Extracting raster above threshold--------------------------#
def extract_per_returnperiods(rainfallfolder, returnperiodfile, outputfolder):
    for days4_file in os.listdir(rainfallfolder):
        if days4_file.endswith(".tif") or days4_file.endswith(".tiff"):
            datename = days4_file.split('.')[2]
            mmname = returnperiodfile.split('_')[5]
            output_filename = 'chirps.indic_extremerain.{0}.{1}.tif'.format(datename, mmname)
            if arcpy.Exists(os.path.join(outputfolder, output_filename)):
                print(output_filename + " is available")
            else:
                arcpy.CheckOutExtension("spatial")
                inRaster1 = Raster(os.path.join(rainfall_folder,days4_file))
                inRaster2 = Raster(returnperiodfile)
                outCon = Con(inRaster1 > inRaster2, inRaster1, 0)
                setnull = SetNull(outCon == 0, outCon)
                setnull.save(os.path.join(outputfolder, output_filename))
                print(output_filename+ " is created")
                arcpy.CheckInExtension("spatial")

#-------------------------Giving Magintude into raster above threshold--------------------------#

def extract_alert_area(crop_folder, threshold_folder, threshold_file, extract_folder):
    for file in os.listdir(crop_folder):
        print(file)
        if file.endswith(".tif"):
            print("extracting magnitude "+file)
            arcpy.CheckOutExtension("spatial")
            multiplier = set_magnitude[threshold_file]
            extract_filename = 'extremerainalert_{0}_{1}.tif'.format(file.split('.')[2], threshold_file.split('_')[5])
            output_greater = GreaterThanEqual(os.path.join(crop_folder, file), os.path.join(threshold_folder,threshold_file))
            output_multiplier = output_greater * multiplier
            output_multiplier.save(os.path.join(extract_folder, extract_filename))
            arcpy.CheckInExtension("spatial")
        print("extracting magnitude " + file+ " done")
    print("processing extremerain Magnitude are done")
    print(".......")
    print(".......")
    print(".......")
    print(".......")


#------------------------- Create File Geodatabase --------------------------#

def createFileGDB(gdb_name, location):
    arcpy.CreateFileGDB_management(location, gdb_name, "CURRENT")

#------------------------- Create Mosaic for raster on the same date --------------------#

def mosaic_alert_area(folder_data, mosaic_folder, gdb_folder):
    filing_data = set()
    sr = arcpy.SpatialReference(4326)
    temp_folder = os.path.join(mosaic_folder, 'temp')
    os.mkdir(temp_folder)
    for file in os.listdir(folder_data):
        if file.endswith(".tif"):
            parseString = file.split('_')
            data_date = parseString[1]
            filing_data.add(data_date)
    for i in filing_data:
        mosaic_files = []
        newfilename = "cli_chirps-v2.0.{0}.extremerainalert.tif".format(i)
        newfilename_gdb = "cli_chirps_v2_0_{0}_extremerainalert".format(i)
        for j in os.listdir(folder_data):
            if j.endswith(".tif"):
                JString = j.split('_')
                if JString[1] == i:
                    mosaic_files.append(os.path.join(folder_data, j))
        mosaic_files.sort()
        arcpy.CheckOutExtension("spatial")
        arcpy.MosaicToNewRaster_management(input_rasters=mosaic_files, output_location=temp_folder,
                                           raster_dataset_name_with_extension=newfilename,
                                           coordinate_system_for_the_raster=sr, pixel_type='4_BIT',
                                           mosaic_method='MAXIMUM',
                                           number_of_bands='1')
        mosaic_file_path = os.path.join(temp_folder, newfilename)
        gdb_mosaic_with_null = os.path.join(gdb_folder, newfilename_gdb)
        set_null_raster = SetNull(mosaic_file_path, mosaic_file_path, "VALUE < 1")
        set_null_raster.save(gdb_mosaic_with_null)
        print(newfilename + " is created")
        arcpy.CheckInExtension("spatial")


#---------------------------Creating CSV Version 2--------------------------#

def final_csv_magnitude_rainfallrate(output_folder):
    print("start processing final CSV......")
    print("creating folder to store csv version 2.....")
    os.mkdir(csv_version_2_folder)
    print("Folder "+csv_version_2_folder+" succesfully created")
    print("start adding column date and threshold into csv file......")
    print("creating folder to store csv version 2.....")
    os.mkdir(csv_version_2_folder_csv)
    print("Folder " + csv_version_2_folder + " succesfully created")
    for i in set_magnitude:
        csv_folder = os.path.join(os.path.join(above_threshold_folder, i),'4_cleaned_csv')
        for csv_file in os.listdir(csv_folder):
            if csv_file.endswith(".csv"):
                csv_file_date = csv_file.split('_')[3]
                csv_file_year = csv_file_date[0:4]
                csv_file_month = csv_file_date[4:6]
                csv_file_day = csv_file_date[6:8]
                csv_file_threshold = csv_file.split('_')[4].strip('.csv')
                csv_file_magnitude = set_magnitude[i]
                a = pd.read_csv(os.path.join(csv_folder,csv_file), sep=',')
                CH_colom = 'r_{0}'.format(csv_file_date)
                a['date'] = csv_file_date
                a['day'] = csv_file_day
                a['month'] = csv_file_month
                a['year'] = csv_file_year
                a['threshold'] = csv_file_threshold
                a['magnitude'] = csv_file_magnitude
                a.rename(columns={CH_colom: "CH"}, inplace=True)
                a.to_csv(os.path.join(csv_version_2_folder_csv, csv_file), index=False, sep=',')
                print(csv_file+ " with date is created")

#-------------------------- Concenating all CSV into one -----------------------------------#

def concenate_csv_files(input_folder):
    filename = 'chirps_rainfall_above_threshold_{0}.csv'.format(year)
    path = input_folder  # use your path
    all_files = glob.glob(
        os.path.join(path, "*.csv"))  # advisable to use os.path.join as this makes concatenation OS independent

    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    concatenated_df.to_csv(os.path.join(csv_version_2_folder,filename), sep=',')
    df = pd.read_csv(os.path.join(csv_version_2_folder, filename), sep=',')
    df = df.loc[df.groupby(['POINT_X', 'POINT_Y']).magnitude.idxmax()]
    df.to_csv(os.path.join(csv_version_2_folder, 'final_'+filename), sep=',')

def rainfall_above_threshold(returnperiod_folder, rainfall_folder, above_threshold_folder):
    print("processing rainfall rate above threshold")
    print("creating folder to save rainfall rate above threshold ")
    os.mkdir(above_threshold_folder)
    for returnperiodfile in os.listdir(returnperiod_folder):
        if returnperiodfile.endswith(".tif") or returnperiodfile.endswith(".tiff"):
            print("processing return period for " + returnperiodfile)
            print("create folder to stored result on this return period...")
            output_folder = os.path.join(above_threshold_folder, returnperiodfile)
            os.mkdir(output_folder)
            print("create folder to stored raster above return period...")
            rasterfolder = os.path.join(output_folder, '1_raster_above_threshold')
            os.mkdir(rasterfolder)
            print("Folder 1_raster_above_threshold is created...")
            print("Start Extracting Data.....")
            extract_per_returnperiods(rainfall_folder, os.path.join(returnperiod_folder, returnperiodfile),
                                      rasterfolder)

            # ----Creating shapefile----#
            # ----Preparing folder to save shapefile----#
            print("create folder to stored shapefile...")
            shapefile_folder = os.path.join(output_folder, '2_shapefile')
            os.mkdir(shapefile_folder)
            print("Folder shapefile is created...")
            print("start processing raster to shapefile.....")

            for raster_file in os.listdir(rasterfolder):
                if raster_file.endswith(".tif") or raster_file.endswith(".tiff"):
                    inRaster = os.path.join(rasterfolder, raster_file)
                    shapefilename = raster_file.split('.')[0] + '_' + raster_file.split('.')[1] + '_' + \
                                    raster_file.split('.')[2] \
                                    + '_' + raster_file.split('.')[3] + '.shp'
                    field = "VALUE"
                    output_point = os.path.join(shapefile_folder, shapefilename)
                    if arcpy.Exists(os.path.join(shapefile_folder, shapefilename)):
                        print("shapefile " + shapefilename + " is available")
                    else:
                        array = arcpy.RasterToNumPyArray(inRaster)
                        if numpy.max(array) > 0:
                            arcpy.RasterToPoint_conversion(inRaster, output_point, field)
                            print("shapefile " + shapefilename + " is created")
                        else:
                            arcpy.CreateFeatureclass_management(shapefile_folder, shapefilename, geometry_type="POINT")
                            print("empty shapefile " + shapefilename + " is created")
                        print("Start adding X and Y coloumn to Shapefiles......")
                        arcpy.AddXY_management(output_point)
                        field_name = 'r_' + raster_file.split('.')[2]
                        field_type = "FLOAT"
                        if arcpy.ListFields(output_point, field_name):
                            print "Field exists"
                        else:
                            arcpy.AddField_management(output_point, field_name, field_type)
                            arcpy.CalculateField_management(output_point, field_name, "!GRID_CODE!", "PYTHON_9.3")
                            # add new x and y point to 3 digit decimal

            print("Creating Shapefile is Done")
            print("create folder to stored csv file...")
            csv_folder = os.path.join(output_folder, '3_shp_to_csv')
            os.mkdir(csv_folder)
            print("Folder shp_to_csv is created...")
            print("start processing shapefile to csv.....")
            arcpy.env.workspace = shapefile_folder
            for i in os.listdir(shapefile_folder):
                if i.endswith(".shp"):
                    print(i)
                    print("processing " + i)
                    new_name = i.split('.')[0]
                    new_name_csv = '{0}.csv'.format(new_name)
                    arcpy.TableToTable_conversion(in_rows=i,
                                                  out_path=csv_folder,
                                                  out_name=new_name_csv)
                    print("csv file " + i + " is created")

            print("Shapefile to csv are completed...")

            print("create folder to stored cleaned csv file...")
            cleaned_folder = os.path.join(output_folder, '4_cleaned_csv')
            os.mkdir(cleaned_folder)
            print("Folder cleaned_csv is created...")
            print("start processing cleaning csv files.....")

            for k in os.listdir(csv_folder):
                if k.endswith(".csv"):
                    b = pd.read_csv(os.path.join(csv_folder, k), sep=',')
                    drop_other_column = b.drop(['OID'], axis=1)
                    drop_other_column.to_csv(os.path.join(cleaned_folder, k), sep=',', index=False)
                    print("cleaned column csv file " + k + " is created")

            print("Cleaning unused coloumn are completed...")
            print("start Merging csv files.....")
            path = cleaned_folder + "/*.csv"
            file_concat = pd.concat(
                [pd.read_csv(f, sep=',').set_index(['POINT_X', 'POINT_Y']) for f in glob.glob(path)],
                axis=1).reset_index()
            file_concat.to_csv(os.path.join(output_folder, "temp_final_output.csv"), index=False, sep=',')
            csv_to_sort = pd.read_csv(os.path.join(output_folder, "temp_final_output.csv"), sep=',')
            final_input_name = 'rainfall_above_threshold_{0}_{1}.csv'.format(returnperiodfile.split('_')[5], year)
            csv_sorted = csv_to_sort.reindex_axis(sorted(csv_to_sort.columns), axis=1)
            csv_sorted.to_csv(os.path.join(above_threshold_folder, final_input_name), sep=',', index=False)
            print("final result is created")
    print("processing rainfall rate above threshold are done")
    print(".......")
    print(".......")
    print(".......")
    print(".......")


rainfall_above_threshold(returnperiod_folder, rainfall_folder, above_threshold_folder)

#=============== processing extremerain magnitude ==============#

print("Start processing extremerain magnitude")
print("........")
print("creating folder to save extremerain magnitude ")
os.mkdir(extremerain_magnitude_folder)
for k in os.listdir(returnperiod_folder):
    if k.endswith(".tif"):
        extract_alert_area(rainfall_folder, returnperiod_folder, k, extremerain_magnitude_folder)


#=============== processing extremerain magnitude ==============#

print("Start processing maximum extremerain magnitude")
print("........")
print("creating folder to save extremerain magnitude ")
gdb_max_extremerain_magnitude_folder = "Max_ExtremeRain_Magnitude"
gdb_max_extremerain_magnitude_folder_gdb = "Max_ExtremeRain_Magnitude.gdb"
os.mkdir(max_extremerain_magnitude_folder)
max_extremerain_magnitude_folder_raster = os.path.join(max_extremerain_magnitude_folder, "1_Raster_Max_FM")
os.mkdir(max_extremerain_magnitude_folder_raster)
arcpy.CreateFileGDB_management(max_extremerain_magnitude_folder_raster, gdb_max_extremerain_magnitude_folder)
gdb_folder = os.path.join(max_extremerain_magnitude_folder_raster, gdb_max_extremerain_magnitude_folder_gdb)
print("start running the script to create raster file...")
mosaic_alert_area(extremerain_magnitude_folder, max_extremerain_magnitude_folder_raster
                  , gdb_folder
                  )

#----Preparing folder to save shapefile----#

print("create folder to stored shapefile...")
max_FM_shapefile_folder = os.path.join(max_extremerain_magnitude_folder,'2_shapefile_Max_FM')
os.mkdir(max_FM_shapefile_folder)
print("Folder shapefile is created...")
print("start processing raster to shapefile.....")
arcpy.env.workspace = gdb_folder
for raster_file in arcpy.ListRasters():
    print("processing "+raster_file+" into shapefile....")
    inRaster = os.path.join(gdb_folder, raster_file)
    shapefilename = raster_file+'.shp'
    field = "VALUE"
    output_point = os.path.join(max_FM_shapefile_folder, shapefilename)
    if arcpy.Exists(os.path.join(max_FM_shapefile_folder, shapefilename)):
        print("shapefile " + shapefilename + " is available")
    else:
        array = arcpy.RasterToNumPyArray(inRaster)
        if numpy.max(array) > 0:
            arcpy.RasterToPoint_conversion(inRaster, output_point, field)
            print("shapefile " + shapefilename + " is created")
        else:
            arcpy.CreateFeatureclass_management(max_FM_shapefile_folder, shapefilename, geometry_type="POINT")
            print("empty shapefile " + shapefilename + " is created")
        print("Start adding X and Y coloumn to Shapefiles......")
        arcpy.AddXY_management(output_point)
        field_name = 'r_' + raster_file.split('_')[4]
        field_type = "FLOAT"
        if arcpy.ListFields(output_point, field_name):
            print "Field exists"
        else:
            arcpy.AddField_management(output_point, field_name, field_type)
            arcpy.CalculateField_management(output_point, field_name, "!GRID_CODE!", "PYTHON_9.3")
            #add new x and y point to 3 digit decimal

print("Creating Shapefile is Done")
print("create folder to stored csv file...")
max_FM_csv_folder = os.path.join(max_extremerain_magnitude_folder,'3_shp_to_csv')
os.mkdir(max_FM_csv_folder)
print("Folder shp_to_csv is created...")
print("start processing shapefile to csv.....")
arcpy.env.workspace = max_FM_shapefile_folder
for i in os.listdir(max_FM_shapefile_folder):
    if i.endswith(".shp"):
        print(i)
        print("processing " + i)
        new_name = i.split('.')[0]
        new_name_csv = '{0}.csv'.format(new_name)
        arcpy.TableToTable_conversion(in_rows=i,
                                      out_path=max_FM_csv_folder,
                                      out_name=new_name_csv)
        print("csv file "+i+" is created")

print("create folder to stored cleaned csv file...")
max_FM_cleaned_folder = os.path.join(max_extremerain_magnitude_folder,'4_cleaned_csv')
os.mkdir(max_FM_cleaned_folder)
print("Folder cleaned_csv is created...")
print("start processing cleaning csv files.....")

for k in os.listdir(max_FM_csv_folder):
    if k.endswith(".csv"):
        b = pd.read_csv(os.path.join(max_FM_csv_folder, k), sep=',')
        drop_other_column = b.drop(['OID'], axis=1)
        drop_other_column.to_csv(os.path.join(max_FM_cleaned_folder, k), sep=',' , index=False)
        print("cleaned column csv file " + k + " is created")

print("Cleaning unused coloumn are completed...")

print("Shapefile to csv are completed...")
print("start Merging csv files.....")
path = max_FM_cleaned_folder+"/*.csv"
file_concat = pd.concat([pd.read_csv(f, sep=',').set_index(['POINT_X', 'POINT_Y']) for f in glob.glob(path)],axis=1).reset_index()
file_concat.to_csv(os.path.join(max_extremerain_magnitude_folder, "temp_final_output.csv"), index=False, sep=',')
csv_to_sort = pd.read_csv(os.path.join(max_extremerain_magnitude_folder, "temp_final_output.csv"), sep=',')
final_input_name = 'max_extremerainmagnitude_{0}.csv'.format(year)
csv_sorted = csv_to_sort.reindex_axis(sorted(csv_to_sort.columns), axis=1)
csv_sorted.to_csv(os.path.join(max_extremerain_magnitude_folder, final_input_name), sep=',', index=False)
print("final result is created")
print("processing rainfall rate above threshold are done")


final_csv_magnitude_rainfallrate(output_output_folder)
concenate_csv_files(csv_version_2_folder_csv)