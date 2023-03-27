# Bias Correction

This script is designed to work with global daily Integrated Multi-satellitE Retrievals for GPM ([IMERG](https://gpm.nasa.gov/data/imerg)) and CPC Unified Gauge-Based Analysis of Global Daily Precipitation ([CPC-UNI](https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html)) data, and compiled as 1 year 1 nc file folowing CPC-UNI structure. 

To do this, you can use Climate Data Operator ([CDO](https://code.mpimet.mpg.de/projects/cdo)) to manipulate the IMERG. Some CDO's module like `mergetime`, `remapbil` and `fillmiss` are useful to merge daily data into annual data then re-grided and clipping the IMERG following CPC-UNI spatial resolution. 

After this steps are done, you can start the correction. Both variables in IMERG and CPC-UNI is written as `precipitationCal` and `precip`, some adjustment are required: parsing filename, directory, variable name, etc.

### Working directory

`/input/imerg` - put your IMERG data here</br>
`/input/cpc` - put your CPC-UNI data here</br>
`/output/{method}/corrected` - location for corrected precipitation output</br>
`/output/{method}/factors` - location for corrected multiplying factors output</br>
`/output/{method}/metrics` - location for corrected statistical metrics output</br>
