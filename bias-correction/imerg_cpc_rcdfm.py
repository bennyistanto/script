# -*- coding: utf-8 -*-
"""
NAME
    imerg_cpc_rcdfm.py
    Bias correction using the Replacement-based CDF Mapping method
DESCRIPTION
    Input data for this script will use GPM IMERG daily data compiled as 1 year 1 nc file
    and CPC Global Unified Gauge-Based Analysis of Daily Precipitation (GUGBADP).
    This script can do a bias correction using the Replacement-based CDF Mapping method
REQUIREMENT
    It required numpy, scipy, pandas and xarray module. So it will work on any machine environment
EXAMPLES
    python imerg_cpc_rcdfm.py
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
import calendar
import xarray as xr
import numpy as np
import scipy.stats as stats
import pandas as pd

def bias_correction(imerg_ds, cpc_ds):
    """
    Correct the bias in the IMERG data using the Replacement-based CDF Mapping method.

    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.

    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.
    """
    # Get the precipitation data from the input datasets
    imerg_precip = imerg_ds['precipitation']
    cpc_precip = cpc_ds['precipitation']

    # Compute the bias-corrected data using the Replacement-based CDF Mapping method
    corrected_precip = np.empty_like(imerg_precip)
    for i in range(imerg_precip.shape[0]):
        for j in range(imerg_precip.shape[1]):
            # Compute the empirical CDFs for the IMERG and CPC data at this location
            imerg_cdf, _ = stats.cumfreq(imerg_precip[i, j, :], numbins=100)
            cpc_cdf, _ = stats.cumfreq(cpc_precip[i, j, :], numbins=100)

            # Map the IMERG data to the CPC CDF using linear interpolation
            corrected_precip[i, j, :] = np.interp(imerg_precip[i, j, :], imerg_cdf, cpc_cdf)

    # Create a new xarray.Dataset with the bias-corrected data
    corrected_ds = xr.Dataset(data_vars={'precipitation': (('time', 'lat', 'lon'), corrected_precip)},
                              coords={'time': imerg_ds['time'],
                                      'lat': imerg_ds['lat'],
                                      'lon': imerg_ds['lon']})
    return corrected_ds

def create_multiplying_factors(corrected_ds, num_dekads=36):
    """
    Create multiplying factors for correcting the IMERG data in the future.

    Parameters:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.
    - num_dekads (int): the number of dekads (10-day periods) to create multiplying factors for.

    Returns:
    - dekad_factors (list of xarray.DataArray): the multiplying factors for each dekad.
    """
    # Compute the number of days in each dekad
    days_per_dekad = corrected_ds.time.size // num_dekads
    # Initialize the list to store the multiplying factors
    dekad_factors = []

    # Loop through each dekad
    for i in range(num_dekads):
        # Get the start and end indices for this dekad
        start_idx = i * days_per_dekad
        end_idx = start_idx + days_per_dekad - 1

        # Get the data for this dekad
        dekad_data = corrected_ds['precipitation'].isel(time=slice(start_idx, end_idx+1))

        # Compute the mean of the data for this dekad
        dekad_mean = dekad_data.mean(dim='time')

        # Divide the mean by the number of days in this dekad, accounting for leap years
        year = corrected_ds['time'][start_idx].dt.year
        month = corrected_ds['time'][start_idx].dt.month
        day = corrected_ds['time'][start_idx].dt.day
        num_days = calendar.monthrange(year, month)[1] - day + 1
        dekad_factor = dekad_mean / num_days

        # Add the multiplying factor for this dekad to the list
        dekad_factors.append(dekad_factor)
    
    return dekad_factors

def calculate_metrics(imerg_ds, cpc_ds):
    """
    Calculate the following metrics for the IMERG and CPC data:
    - relative bias
    - Pearson correlation coefficient
    - root mean squared error
    - mean absolute error
    - probability of detection
    - false alarm ratio
    - critical success index

    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data.
    - cpc_ds (xarray.Dataset): the CPC data.

    Returns:
    - metrics (pandas.DataFrame): a dataframe containing the metric values.
    """
    # Get the precipitation data from the input datasets
    imerg_precip = imerg_ds['precipitation']
    cpc_precip = cpc_ds['precipitation']

    # Calculate the relative bias
    relative_bias = (imerg_precip.sum(dim='time') / cpc_precip.sum(dim='time')).mean(dim=('lat', 'lon'))

    # Calculate the Pearson correlation coefficient
    pearson = imerg_precip.corr(cpc_precip, dim='time')

    # Calculate the root mean squared error
    rmse = (((imerg_precip - cpc_precip)**2).mean(dim='time')**0.5).mean(dim=('lat', 'lon'))

    # Calculate the mean absolute error
    mae = (np.abs(imerg_precip - cpc_precip)).mean(dim='time').mean(dim=('lat', 'lon'))

    # Calculate the probability of detection
    pod = (imerg_precip > 0).sum(dim='time') / (cpc_precip > 0).sum(dim='time')

    # Calculate the false alarm ratio
    far = ((imerg_precip > 0) & (cpc_precip == 0)).sum(dim='time') / (cpc_precip == 0).sum(dim='time')

    # Calculate the critical success index
    csi = pod / (pod + far)

    # Create a dataframe to store the metric values
    metrics = pd.DataFrame({'relative_bias': relative_bias,
                            'pearson': pearson,
                            'rmse': rmse,
                            'mae': mae,
                            'pod': pod,
                            'far': far,
                            'csi': csi})
    return metrics

def main():
    # Load the IMERG data
    imerg_ds = xr.open_mfdataset('imerg_*.nc')

    # Load the CPC data
    cpc_ds = xr.open_mfdataset('cpc_*.nc')

    # Initialize an empty dataframe to store the metric values
    metrics = pd.DataFrame(columns=['relative_bias', 'pearson', 'rmse', 'mae', 'pod', 'far', 'csi'])

    # Loop through the annual data
    for year in range(2001, 2023):
        # Get the data for this year
        imerg_year_ds = imerg_ds.sel(time=imerg_ds['time.year'] == year)
        cpc_year_ds = cpc_ds.sel(time=cpc_ds['time.year'] == year)

        # Correct the bias in the IMERG data
        corrected_ds = bias_correction(imerg_year_ds, cpc_year_ds)

        # Save the corrected data to a NetCDF file
        corrected_ds.to_netcdf(f'corrected_{year}.nc')
        
        # Calculate the metrics for the corrected data
        year_metrics = calculate_metrics(corrected_ds, cpc_year_ds)

        # Add the metric values for this year to the dataframe
        metrics = metrics.append(year_metrics, ignore_index=True)

        # Output the metric values to a CSV file
    metrics.to_csv('metrics.csv', index=False)

    # Create the multiplying factor files
    dekad_factors = create_multiplying_factors(corrected_ds)
    for i, factor in enumerate(dekad_factors):
        factor.to_netcdf(f'multiplying_factor_{i+1}.nc')

if __name__ == '__main__':
    main()