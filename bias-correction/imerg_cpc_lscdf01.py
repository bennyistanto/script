# -*- coding: utf-8 -*-
"""
NAME
    imerg_cpc_lscdf01.py
    Bias correction using LSCDF
DESCRIPTION
    Input data for this script will use GPM IMERG daily data compiled as 1 year 1 nc file
    and CPC Global Unified Gauge-Based Analysis of Daily Precipitation (GUGBADP).
    This script can do a bias correction using the mean-based Linear Scaling (LS) and 
    quantile-mapping Cumulative Distribution Function (CDF) matching approaches (LSCDF)
REQUIREMENT
    It required numpy and xarray module. So it will work on any machine environment
EXAMPLES
    python imerg_cpc_lscdf01.py
NOTES
    This script is designed to work with global daily IMERG and GUGBADP data, and compiled as
    1 year 1 nc file folowing GUGBDAP structure. To do this, you can use Climate Data Operator 
    (CDO) to manipulate the IMERG. Some CDO's module like mergetime and remapbil are useful to 
    merge daily data into annual data then re-grided following GUGBDAP spatial resolution. 
    After this steps are done, you can start the correction.
    If using other data, some adjustment are required: parsing filename, directory, etc.
CONTACT
    Benny Istanto
    Climate Geographer
    GOST, The World Bank
LICENSE
    This script is in the public domain, free from copyrights or restrictions.
VERSION
    $Id$
TODO
    xx
"""
import xarray as xr
import numpy as np

def bias_correction(imerg, cpc):
    """
    Perform bias correction on imerg using cpc as reference.
    """
    # Calculate mean and standard deviation of cpc
    cpc_mean = cpc.mean()
    cpc_std = cpc.std()
    
    # Perform LS bias correction
    imerg_ls = (imerg - imerg.mean()) * cpc_std / imerg.std() + cpc_mean
    
    # Perform LSCDF bias correction
    # First, calculate the CDF of both imerg and cpc
    imerg_cdf = imerg.rank(pct=True)
    cpc_cdf = cpc.rank(pct=True)
    # Then, interpolate the CDF of imerg onto the values of cpc
    imerg_lscdf = imerg_cdf.interp(precip=cpc)
    # Finally, invert the CDF to get the corrected values of imerg
    imerg_lscdf = xr.DataArray(np.interp(imerg_lscdf, cpc_cdf, cpc), dims=['lat', 'lon'])
    
    return imerg_ls, imerg_lscdf

def calculate_metrics(corrected, cpc):
    """
    Calculate relative bias, Pearson correlation, RMSE, MAE, POD, FAR, and CSI 
    for corrected data compared to cpc.
    """
    relative_bias = (corrected.mean() - cpc.mean()) / cpc.mean()
    pearson = xr.corr(corrected, cpc, dim='time')
    rmse = np.sqrt(((corrected - cpc) ** 2).mean())
    mae = (corrected - cpc).abs().mean()
    
    # Calculate POD, FAR, and CSI
    pod = ((corrected > 0) & (cpc > 0)).sum() / (cpc > 0).sum()
    far = ((corrected == 0) & (cpc > 0)).sum() / (cpc > 0).sum()
    csi = pod / (pod + far)
    
    return relative_bias, pearson, rmse, mae, pod, far, csi

# Loop over years and perform bias correction
for year in range(2001, 2022):
    # Load the imerg and cpc data for the current year
    imerg = xr.open_dataset(f'imerg_{year}.nc')['precip']
    cpc = xr.open_dataset(f'cpc_{year}.nc')['precip']
    
    # Perform bias correction and calculate metrics
    imerg_ls, imerg_lscdf = bias_correction(imerg, cpc)
    metrics_ls = calculate_metrics(imerg_ls, cpc)
    metrics_lscdf = calculate_metrics(imerg_lscdf, cpc)
    
    # Save the corrected data to NetCDF files
    imerg_ls.to_netcdf(f'imerg_ls_{year}.nc')
    imerg_lscdf.to_netcdf(f'imerg_lscdf_{year}.nc')
    
    # Write the metrics to a CSV file
    with open('metrics.csv', 'a') as f:
        f.write(f'{year},LS,{metrics_ls[0]},{metrics_ls[1]},{metrics_ls[2]},{metrics_ls[3]},\
            {metrics_ls[4]},{metrics_ls[5]},{metrics_ls[6]}\n')
        f.write(f'{year},LSCDF,{metrics_lscdf[0]},{metrics_lscdf[1]},{metrics_lscdf[2]},{metrics_lscdf[3]},\
            {metrics_lscdf[4]},{metrics_lscdf[5]},{metrics_lscdf[6]}\n')

# This code will load the corrected data for all years, calculate the multiplying factor for 
# each 10-day period, and save the resulting multiplying factor to NetCDF files. The multiplying 
# factor will be calculated by taking the mean of the corrected data for each 10-day period and 
# dividing it by the overall mean of the corrected data.

# Load the corrected data for all years
imerg_ls_all = xr.concat([xr.open_dataset(f'imerg_ls_{year}.nc')['precip'] for year in range(2001, 2022)], dim='time')
imerg_lscdf_all = xr.concat([xr.open_dataset(f'imerg_lscdf_{year}.nc')['precip'] for year in range(2001, 2022)], dim='time')

# Calculate the multiplying factor for each 10-day period
imerg_ls_factor = imerg_ls_all.groupby('time.time').mean() / imerg_ls_all.mean()
imerg_lscdf_factor = imerg_lscdf_all.groupby('time.time').mean() / imerg_lscdf_all.mean()

# Divide the multiplying factor by the number of days in each 10-day period
# Use the number of days in each month to determine the number of days in each 10-day period, 
# accounting for leap years
num_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
imerg_ls_factor = imerg_ls_factor / (num_days[imerg_ls_factor.time.dt.month - 1] / 3)
imerg_lscdf_factor = imerg_lscdf_factor / (num_days[imerg_lscdf_factor.time.dt.month - 1] / 3)
imerg_ls_factor.where(imerg_ls_factor.time.dt.is_leap_year, imerg_ls_factor / \
    (num_days[imerg_ls_factor.time.dt.month - 1] / 3 - 1), inplace=True)
imerg_lscdf_factor.where(imerg_lscdf_factor.time.dt.is_leap_year, imerg_lscdf_factor / \
    (num_days[imerg_lscdf_factor.time.dt.month - 1] / 3 - 1), inplace=True)

# Save the multiplying factor to separate NetCDF files for each 10-day period
for i, time in enumerate(imerg_ls_factor.time):
    imerg_ls_factor.sel(time=time).to_netcdf(f'imerg_ls_factor_{i+1:02d}.nc')
    imerg_lscdf_factor.sel(time=time).to_netcdf(f'imerg_lscdf_factor_{i+1:02d}.nc')

# You can then use these multiplying factor files to correct the IMERG data after 2022 
# by multiplying the IMERG data by the appropriate multiplying factor. For example, to correct 
# the IMERG data for the first 10-day period of 2023 using the LS method, you could do the following:

# Load the multiplying factor and IMERG data for the first 10-day period of 2023
imerg_ls_factor = xr.open_dataset('imerg_ls_factor_01.nc')['precip']
imerg_2023_1 = xr.open_dataset('imerg_2023_1.nc')['precip']

# Correct the IMERG data using the multiplying factor
imerg_2023_1_corrected = imerg_2023_1 * imerg_ls_factor