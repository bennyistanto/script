import arcpy  
import os  
import math  
  
# get SA license  
arcpy.CheckOutExtension("Spatial")  
  
# settings, input data  
max_dist = 12000  
steps = 360 # 1 point each degree  
ws = r"Z:\Temp\SurfaceBuffer\Bali_UTM.gdb"  
dem = r"Z:\Temp\SurfaceBuffer\Bali_UTM.gdb\DEM"  
fc_pnt= r"Z:\Temp\SurfaceBuffer\Bali_UTM.gdb\Mt_Agung"  
  
# output and intermediate names  
slp_per_name = "slope_perc"  
fc_buf_name = "pnt_buf"  
fc_linepnt_name = "rad_pnt"  
fc_linepntdem_name = "rad_pnt_dem"  
fc_linepntdemsel_name = "rad_pnt_dem_sel"  
fc_buf3D_name = "buf_3D"  
  
# internal settings  
arcpy.env.workspace = ws  
arcpy.env.overwriteOutput = True  
# ws_mem = "IN_MEMORY"  
fld_rasval = "RASTERVALU"  
  
# use cellsize as interval  
##ras_dem = arcpy.Raster(dem)  
##cellw = ras_dem.meanCellWidth  
##cellh = ras_dem.meanCellHeight  
##pixsize = int((cellw + cellh) / 2)  
  
#overwrite precision along line  
pixsize = 30.92336053
steps2 = max_dist / pixsize  
  
# Get point from input fc  
with arcpy.da.SearchCursor(fc_pnt, ("SHAPE@XY")) as curs:  
    for row in curs:  
        pnt_cc = arcpy.Point(row[0][0],row[0][1])  
        break  
  
# Createa normal buffer on point with max distance  
fc_buf = os.path.join(ws, fc_buf_name)  
arcpy.Buffer_analysis(fc_pnt, fc_buf, "{0} METERS".format(max_dist))  
  
# get buffer geometry  
with arcpy.da.SearchCursor(fc_buf, ("SHAPE@")) as curs:  
    for row in curs:  
        geom = row[0]  
        break  
  
# get outline  
polyline = geom.boundary()  
length = polyline.length  
interval = length / steps  
  
# create new featureclass to hold points on lines  
sr = arcpy.Describe(fc_pnt).spatialReference  
arcpy.CreateFeatureclass_management(ws, fc_linepnt_name, "POINT", "", "DISABLED", "DISABLED", sr)  
fc_linepnt = os.path.join(ws, fc_linepnt_name)  
  
# add columns  
fld_id = "LineID"  
fld_dist = "Distance"  
fld_slpper = "SlopePerc"  
fld_dist3D = "Dist3D"  
fld_id2 = "LinePntID"  
  
arcpy.AddField_management(fc_linepnt, fld_id, "LONG")  
arcpy.AddField_management(fc_linepnt, fld_dist, "DOUBLE")  
arcpy.AddField_management(fc_linepnt, fld_slpper, "DOUBLE")  
arcpy.AddField_management(fc_linepnt, fld_dist3D, "DOUBLE")  
arcpy.AddField_management(fc_linepnt, fld_id2, "TEXT", 25)  
  
flds = ("SHAPE@", fld_id, fld_dist, fld_id2)  
with arcpy.da.InsertCursor(fc_linepnt, flds) as curs:  
    # loop over boundary  
    for i in range(0, steps):  
        d = i * interval  
        line_id = i  
  
        # extract point on buffer outline  
        pnt = polyline.positionAlongLine(d, False)  
  
        # create line from center to pnt on line  
        arr = arcpy.Array()  
        arr.removeAll  
        arr.add(pnt_cc)  
        arr.add(pnt.firstPoint)  
        line = arcpy.Polyline(arr)  
  
        # extract points on line  
        for j in range(0, int(steps2)):  
            dist = int(j * pixsize)  
            pnt2 = line.positionAlongLine(dist, False)  
  
            # add to fc in mem with line ID and eucledian distance  
            val_id = "{0}_{1}".format(line_id, dist)  
            curs.insertRow((pnt2, line_id, dist, val_id))  
  
  
# extract DEM values  
fc_linepntdem = os.path.join(ws, fc_linepntdem_name)  
arcpy.sa.ExtractValuesToPoints(fc_linepnt, dem, fc_linepntdem, "INTERPOLATE", "VALUE_ONLY")  
  
# create nested dictionary for update  
##flds = ";".join([fld_id, fld_dist, fld_rasval, fld_slpper, fld_dist3D])  
##rows = arcpy.SearchCursor(fc_linepntdem, fields=flds, sort_fields="{0} A; {1} A".format(fld_id, fld_dist))  
##dct_vals = {"{0}_{1}".format(r.getValue(fld_id), r.getValue(fld_dist)):[r.getValue(fld_id), r.getValue(fld_dist), r.getValue(fld_rasval), -1, -1] for r in rows}  
flds = (fld_id, fld_dist, fld_rasval, fld_id2)  
dct_vals = {r[3]:[r[0], r[1], r[2], -1, -1] for r in arcpy.da.SearchCursor(fc_linepntdem, flds)}  
  
# create list of sort order data  
flds = ";".join([fld_id, fld_dist, fld_id2])  
rows = arcpy.SearchCursor(fc_linepntdem, fields=flds, sort_fields="{0} A; {1} A".format(fld_id, fld_dist))  
lst_vals_ids = [r.getValue(fld_id2) for r in rows]  
  
# save for each line the id of the outside point  
dct_res = {}  
# do some magic...  
for val_id in lst_vals_ids:  
    if val_id in dct_vals:  
        lst = dct_vals[val_id]  
        line_id = lst[0]  
        if lst[1] == 0:  
            # start of line  
            lst[3] = 0  
            lst[4] = 0  
            h0 = lst[2]  
            dct_vals[val_id] = lst  
            distcum = 0  
        else:  
            # other points  
            h1 = lst[2]  
            h_dif = abs(h1-h0)  
            dist3D = math.sqrt((h_dif**2) + (pixsize**2))  
            slope = h_dif * 100.0 / pixsize  
            distcum += dist3D  
            lst[3] = slope  
            lst[4] = distcum  
            dct_vals[val_id] = lst  
            h0 = h1  
            if distcum <= max_dist:  
                dct_res[line_id] = val_id  
  
# write slope and distance cum to featureclass  
flds = (fld_id, fld_dist, fld_slpper, fld_dist3D)  
with arcpy.da.UpdateCursor(fc_linepntdem, flds) as curs:  
    for row in curs:  
        val_id = "{0}_{1}".format(row[0], row[1])  
        if val_id in dct_vals:  
            lst = dct_vals[val_id]  
            row[2] = lst[3]  
            row[3] = lst[4]  
            curs.updateRow(row)  
  
# select points within max distance  
arcpy.MakeFeatureLayer_management(fc_linepntdem, "lyr")  
  
# define where clause based on list of id's  
txt_ids = (str(dct_res.values()))[1:-1]  
txt_ids = txt_ids.replace('u','')  
where = "{0} IN ({1})".format(arcpy.AddFieldDelimiters(fc_linepntdem, fld_id2), txt_ids)  
arcpy.SelectLayerByAttribute_management("lyr", "NEW_SELECTION", where)  
  
# copy to new fc (for debug)  
fc_linepntdemsel = os.path.join(ws, fc_linepntdemsel_name)  
arcpy.CopyFeatures_management("lyr", fc_linepntdemsel)  
  
# create convexHull on points for 3D buffer  
# this will give erroneous results on concave parts  
##cnt = 0  
##with arcpy.da.SearchCursor(fc_linepntdemsel, ("SHAPE@")) as curs:  
##    for row in curs:  
##        if cnt == 0:  
##            pntg = row[0]  
##        else:  
##            pntg_tmp = row[0]  
##            pntg = pntg.union(pntg_tmp)  
##        cnt += 1  
##  
##pol = pntg.convexHull()  
  
# create polygon manually, sort points on line_id  
arr_pol = arcpy.Array()  
fld_shape = arcpy.Describe(fc_linepntdemsel).shapeFieldName
flds = ";".join([fld_id, fld_shape])  
rows = arcpy.SearchCursor(fc_linepntdemsel, fields=flds, sort_fields="{0} A".format(fld_id))  
cnt = 0  
for row in rows:  
    if cnt == 0:  
        end_pntg = row.getValue(fld_shape)  
        end_pnt = end_pntg.firstPoint  
  
    pntg = row.getValue(fld_shape)  
    pnt = pntg.firstPoint  
    arr_pol.add(pnt)  
    cnt += 1  
  
arr_pol.add(end_pnt)  
del row  
del rows  
  
pol = arcpy.Polygon(arr_pol)  
  
# write polygon to output fc  
fc_buf3D = os.path.join(ws, fc_buf3D_name)  
arcpy.CreateFeatureclass_management(ws, fc_buf3D_name, "POLYGON", "", "DISABLED", "DISABLED", sr)  
with arcpy.da.InsertCursor(fc_buf3D, "SHAPE@") as curs:  
    row = (pol,)  
    curs.insertRow(row)  
  
# return SA license  
arcpy.CheckInExtension("Spatial")
