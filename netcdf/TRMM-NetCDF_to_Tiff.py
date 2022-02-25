#!/usr/bin/env python
"""
SYNOPSIS
    NetCDF_to_Tiff [-h,--help] [-v,--verbose] [-i,--input] [-o, --output]
DESCRIPTION
    This script converts a NetCDF time-series to a set
	of TIFF images, one for each time step. It is designed 
	for converting TRMM data.
EXAMPLES
    python NetCDF_to_Tiff.py -i d:/data/3B42RT_daily.nc -o d:/data/3B42RT_daily/
EXIT STATUS
    TODO: List exit codes
AUTHOR
    Rochelle O'Hagan <spatialexplore@gmail.com>
LICENSE
    This script is in the public domain, free from copyrights or restrictions.
VERSION
    $Id$
"""
import arcpy
import os, optparse, time, sys, traceback


def main(in_netcdf, out_dir):
    try:
        nc_fp = arcpy.NetCDFFileProperties(in_netcdf)
        nc_dim = 'time'
        in_netcdf_layer = os.path.basename(in_netcdf).split('.')[0]
        out_layer = "{0}_layer".format(in_netcdf_layer)
        arcpy.MakeNetCDFRasterLayer_md(in_netcdf, "r", "lon", "lat", out_layer, "", "", "")
        print "Created NetCDF Layer for " + out_layer

        for i in range(0, nc_fp.getDimensionSize(nc_dim)):
            nc_dim_value = nc_fp.getDimensionValue(nc_dim, i)
            print("\tDimension value: {0}".format(nc_dim_value))
            print("\tDimension index: {0}".format(nc_fp.getDimensionIndex(nc_dim, nc_dim_value)))
            dmy = nc_dim_value.split('/')
            date_str = "{2}.{1}.{0}".format(dmy[0], dmy[1], dmy[2])
            out_layer_file = "{0}{1}.{2}.tif".format(out_dir, in_netcdf_layer, date_str)

            # Execute SelectByDimension tool
            valueSelect = ["time", i]
            net_layer = arcpy.SelectByDimension_md(out_layer, [valueSelect], "BY_INDEX")
            #define output tif name
            arcpy.CopyRaster_management(net_layer, out_layer_file)
            arcpy.AddMessage(out_layer_file + " " "exported" + " " + "successfully")
    except Exception as err:
        print(err)

if __name__ == '__main__':
    try:
        start_time = time.time()
        parser = optparse.OptionParser(formatter=optparse.TitledHelpFormatter(), usage=globals()['__doc__'], version='$Id$')
        parser.add_option ('-v', '--verbose', action='store_true', default=False, help='verbose output')
        parser.add_option ('-i', '--input', dest='netcdf_file', action='store', help='netcdf filename')
        parser.add_option ('-o', '--output', dest='output_dir', action='store', help='output directory')
        (options, args) = parser.parse_args()
        #if len(args) < 1:
        #    parser.error ('missing argument')
        if options.verbose: print time.asctime()
        input_f = "r:/TRMM/3b42rt_daily.nc" # Default value - should be set using options
        if options.netcdf_file:
            input_f = options.netcdf_file
            print 'netcdf file=', input_f
        output_d = "r:/TRMM/test2/" # Default value - should be set using options
        if options.output_dir:
            output_d = options.output_dir
            print 'output directory=', output_d
        main(input_f, output_d)
        if options.verbose: print time.asctime()
        if options.verbose: print 'TOTAL TIME IN MINUTES:',
        if options.verbose: print (time.time() - start_time) / 60.0
        sys.exit(0)
    except KeyboardInterrupt, e: # Ctrl-C
        raise e
    except SystemExit, e: # sys.exit()
        raise e
    except Exception, e:
        print 'ERROR, UNEXPECTED EXCEPTION'
        print str(e)
        traceback.print_exc()
        os._exit(1)
