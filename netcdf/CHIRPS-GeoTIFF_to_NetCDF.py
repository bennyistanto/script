#!/usr/bin/env python
"""
-------------------------------------------------------------------------------------------------------------
Convert CHIRPS GeoTIFF in a folder to single NetCDF file with time dimension enabled that is CF-Compliant
http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html
 
Based on Rich Signell's answer on StackExchange: https://gis.stackexchange.com/a/70487
This script was tested using CHIRPS dekad data. Adjustment is needed if using other timesteps data for CHIRPS
NCO (http://nco.sourceforge.net) must be installed before using this script
 
Modified by
Benny Istanto, UN World Food Programme, benny.istanto@wfp.org
-------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import datetime as dt
import os
import gdal
import netCDF4
import re

ds = gdal.Open('/path/to/dir/chirps-v2.0.1981.01.1.tif') # Data location
a = ds.ReadAsArray()
nlat,nlon = np.shape(a)

b = ds.GetGeoTransform() #bbox, interval
lon = np.arange(nlon)*b[1]+b[0]
lat = np.arange(nlat)*b[5]+b[3]

basedate = dt.datetime(1980,1,1,0,0,0)


# Create NetCDF file
nco = netCDF4.Dataset('chirps_dekads.nc','w',clobber=True) # Output name


# Chunking is optional, but can improve access a lot: 
# (see: http://www.unidata.ucar.edu/blogs/developer/entry/chunking_data_choosing_shapes)
chunk_lon=10
chunk_lat=10
chunk_time=12


# Create dimensions, variables and attributes:
nco.createDimension('lon',nlon)
nco.createDimension('lat',nlat)
nco.createDimension('time',None)

timeo = nco.createVariable('time','f4',('time'))
timeo.units = 'days since 1980-1-1 00:00:00'
timeo.standard_name = 'time'
timeo.calendar = 'gregorian'
timeo.axis = 'T'

lono = nco.createVariable('lon','f4',('lon'))
lono.units = 'degrees_east'
lono.standard_name = 'longitude'
lono.long_name = 'longitude'
lono.axis = 'X'

lato = nco.createVariable('lat','f4',('lat'))
lato.units = 'degrees_north'
lato.standard_name = 'latitude'
lato.long_name = 'latitude'
lato.axis = 'Y'

# Create container variable for CRS: lon/lat WGS84 datum
crso = nco.createVariable('crs','i4')
crso.long_name = 'Lon/Lat Coords in WGS84'
crso.grid_mapping_name='latitude_longitude'
crso.longitude_of_prime_meridian = 0.0
crso.semi_major_axis = 6378137.0
crso.inverse_flattening = 298.257223563

# Create float variable for precipitation data, with chunking
pcpo = nco.createVariable('precip', 'f4',  ('time', 'lat', 'lon'), 
   zlib=True,chunksizes=[chunk_time,chunk_lat,chunk_lon],fill_value=-9999.)
pcpo.units = 'mm'
pcpo.standard_name = 'convective precipitation rate'
pcpo.long_name = 'Climate Hazards group InfraRed Precipitation with Stations'
pcpo.time_step = 'dekad'
pcpo.missing_value = -9999.
pcpo.geospatial_lat_min = -50.
pcpo.geospatial_lat_max = 50.
pcpo.geospatial_lon_min = -180.
pcpo.geospatial_lon_max = 180.
pcpo.grid_mapping = 'crs'
pcpo.set_auto_maskandscale(False)

# Additional attributes
nco.Conventions='CF-1.6'
nco.title = "CHIRPS v2.0"
nco.history = "created by Climate Hazards Group. University of California at Santa Barbara"
nco.version = "Version 2.0"
nco.comments = "time variable denotes the first day of the given dekad."
nco.website = "https://www.chc.ucsb.edu/data/chirps"
nco.date_created = "2020-12-19"
nco.creator_name = "Benny Istanto"
nco.creator_email = "benny.istanto@wfp.org"
nco.institution = "UN World Food Programme"
nco.note = "The data is developed to support regular updating procedure for SPI analysis (https://github.com/wfpidn/SPI). This activities will support WFP to assess extreme dry and wet periods as part of WFP's Seasonal Monitoring"


# Write lon,lat
lono[:]=lon
lato[:]=lat

pat = re.compile('chirps-v2.0.[0-9]{4}\.[0-9]{2}\.[0-9]{1}')
itime=0

# Step through data, writing time and data to NetCDF
for root, dirs, files in os.walk('/Users/bennyistanto/Temp/CHIRPS/Regular/Dekad/Rain/'):
    dirs.sort()
    files.sort()
    for f in files:
        if re.match(pat,f):
            # read the time values by parsing the filename
            year=int(f[12:16])
            mon=int(f[17:19])
            dekad=int(f[20:21])
            date=dt.datetime(year,mon,dekad,0,0,0)
            print(date)
            dtime=(date-basedate).total_seconds()/86400.
            timeo[itime]=dtime
           # precipitation
            pcp_path = os.path.join(root,f)
            print(pcp_path)
            pcp=gdal.Open(pcp_path)
            a=pcp.ReadAsArray()  #data
            pcpo[itime,:,:]=a
            itime=itime+1

nco.close()
