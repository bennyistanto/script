# Open the netcdf file
nc <- nc_open(filename)
nc
File testMultiDim.nc (NC_FORMAT_NETCDF4):

     2 variables (excluding dimension variables):
        float precipitation[longitude,latitude,Time]   (Chunking: [1080,450,1])  (Compression: level 9)
            units: mm/month
            _FillValue: -999
            long_name: Monthly_Total_Precipitation
        float temperature[longitude,latitude,Time]   (Chunking: [1080,450,1])  (Compression: level 9)
            units: degC
            _FillValue: -999
            long_name: Monthly_Avg_Temperature_degC

     3 dimensions:
        longitude  Size:2160
            units: degrees_east
            long_name: longitude
        latitude  Size:900
            units: degrees_north
            long_name: latitude
        Time  Size:12   *** is unlimited ***
            units: months
            long_name: Month_of_year

    4 global attributes:
        Title: MultiDimesionsalNCDFTest
        Source: Some example data from the raster package
        References: See the raster package
        Created on: Wed Nov 28 10:35:53 2018
