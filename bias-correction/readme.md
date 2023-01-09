# Bias Correction

This script is designed to work with global daily Integrated Multi-satellitE Retrievals for GPM ([IMERG](https://gpm.nasa.gov/data/imerg)) and CPC Global Unified Gauge-Based Analysis of Daily Precipitation ([GUGBADP](https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html)) data, and compiled as 1 year 1 nc file folowing GUGBDAP structure. 

To do this, you can use Climate Data Operator ([CDO](https://code.mpimet.mpg.de/projects/cdo)) to manipulate the IMERG. Some CDO's module like `mergetime` and `remapbil` are useful to merge daily data into annual data then re-grided the IMERG following GUGBDAP spatial resolution. 
You can clip the NetCDF file based on area of interest using a shapefile.

After this steps are done, you can start the correction. If using other data, some adjustment are required: parsing filename, directory, etc.
