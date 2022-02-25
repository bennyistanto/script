def main():
    print "load modules..."
    import arcpy
    arcpy.env.overwriteOutput = True

    import numpy as np
    template1 = r'D:\UJI_REGRESI\sst_anomaly\sst_{}.tif'
    template2 = r'D:\UJI_REGRESI\precip_anomaly\precip_anom_{}.tif'
    nodata = -3.4028235e+38
    out_ras = r'C:\UJI_REGRESI\correlation.tif'

    print "create nested numpy array list..."
    lst_np_ras = []
    for i in range(0, 444):
        ras_path1 = template1.format("%01d" % (i,))
        print " - ", ras_path1
        ras_np1 = arcpy.RasterToNumPyArray(ras_path1)
        ras_path2 = template2.format("%01d" % (i,))
        print " - ", ras_path2
        ras_np2 = arcpy.RasterToNumPyArray(ras_path2)
        lst_np_ras.append([ras_np1, ras_np2])

    print "read props numpy raster..."
    ras_np = lst_np_ras[0][0]  # take first numpy array from list
    rows = ras_np.shape[0]
    cols = ras_np.shape[1]
    print " - rows:", rows
    print " - cols:", cols

    print "create output numpy array..."
    ras_path = template1.format("%01d" % (1,))
    raster = arcpy.Raster(ras_path)
    ras_np_res = np.ndarray((rows, cols))
    print " - out rows:", ras_np_res.shape[0]
    print " - out cols:", ras_np_res.shape[1]

    print "loop through pixels..."
    pix_cnt = 0
    for row in range(rows):
        for col in range(cols):
            pix_cnt += 1
            if pix_cnt % 5000 == 0:
                print " - row:", row, "  col:", col, "  pixel:", pix_cnt
            lst_vals1 = []
            lst_vals2 = []
            try:
                for lst_pars in lst_np_ras:
                    lst_vals1.append(lst_pars[0][row, col])
                    lst_vals2.append(lst_pars[1][row, col])
                lst_vals1 = ReplaceNoData(lst_vals1, nodata)
                lst_vals2 = ReplaceNoData(lst_vals2, nodata)
                # perform calculation on list
                correlation = CalculateCorrelation(lst_vals1, lst_vals2, nodata)
                ras_np_res[row, col] = correlation
            except Exception as e:
                print "ERR:", e
                print " - row:", row, "  col:", col, "  pixel:", pix_cnt
                print " - lst_vals1:", lst_vals1
                print " - lst_vals2:", lst_vals2

    pnt = arcpy.Point(raster.extent.XMin, raster.extent.YMin) #  - raster.meanCellHeight
    xcellsize = raster.meanCellWidth
    ycellsize = raster.meanCellHeight

    print "Write output raster..."
    print " - ", out_ras
    ras_res = arcpy.NumPyArrayToRaster(ras_np_res, lower_left_corner=pnt, x_cell_size=xcellsize,
                                 y_cell_size=ycellsize, value_to_nodata=nodata)
    ras_res.save(out_ras)
    arcpy.DefineProjection_management(in_dataset=out_ras, coor_system="GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]]")


def CalculateCorrelation(a, b, nodata):
    import numpy
    try:
        coef = numpy.corrcoef(a,b)
        return coef[0][1]
    except:
        return nodata


def ReplaceNoData(lst, nodata):
    res = []
    for a in lst:
        if a < nodata / 2.0:
            res.append(None)
        else:
            res.append(a)
    return res


if __name__ == '__main__':
    main()