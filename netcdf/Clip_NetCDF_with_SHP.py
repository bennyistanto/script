#!/usr/bin/python
# Source: https://gis.stackexchange.com/a/354798/97103

import geopandas
import rioxarray
import xarray
from shapely.geometry import mapping


CHIRPS_daily = xarray.open_dataarray('Z:\Temp\CHIRPS\NC\chirps-v2.0.1981.days_p05.nc')
CHIRPS_daily.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
CHIRPS_daily.rio.write_crs("epsg:4326", inplace=True)
Shapefile = geopandas.read_file('Z:\Temp\CHIRPS\Subset\idn_bnd_adm2_lite_chirps_diss_a.shp', crs="epsg:4326")

clipped = CHIRPS_daily.rio.clip(Shapefile.geometry.apply(mapping), Shapefile.crs, drop=False)