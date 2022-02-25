# -*- coding: utf-8 -*-
import sys
import numpy as np
import arcpy

arcpy.env.overwriteOutput = True

ft = {'bool': lambda x: repr(x.astype(np.int32)),
      'float_kind': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=120, precision=2, suppress=True,
                    threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')  # change to a single -

script = sys.argv[0]  # print this should you need to locate the script

# ---- You need to specify the folder where the tif files reside ------------
src_flder = r"Z:\Temp\CHIRPS\Daily\Data\1995" # just change the year
out_flder =  r"Z:\Temp\CHIRPS\Daily\Data" # make a result folder to put stuff
out_year = src_flder.split("\\")[-1]
max_name = "{}\\chirps-v.2.0.{}.1days.max.tif".format(out_flder, out_year)

# ---- Process time ----
arcpy.env.workspace = src_flder
rasters = arcpy.ListRasters()
arrs = []
for i in rasters:
    r = arcpy.RasterToNumPyArray(i, nodata_to_value=0.0)
    arrs.append(r)
# ---- Convert to an integer dtype since the float precision is a bit much
a = np.array(arrs)
m = np.where(a > 0., 1, 0)
a_m = a * m
a_m = np.ndarray.astype(a_m, np.int32)
a_max = np.max(a_m, axis=0)

out_max = arcpy.NumPyArrayToRaster(a_max,
                                   arcpy.Point(-180,-50),
                                   x_cell_size=0.050000000745058,
                                   y_cell_size=0.050000000745058,
                                   value_to_nodata=0)
out_max.save(max_name)

del a, m, a_m, a_max, max_name, out_max