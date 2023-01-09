# -*- coding: utf-8 -*-
"""
NAME
    imerg_cpc_idfcurves.py
    Bias correction using Intensity-Duraton-Frequency (IDF) curves correction method
DESCRIPTION
    Input data for this script will use GPM IMERG daily data compiled as 1 year 1 nc file
    and CPC Global Unified Gauge-Based Analysis of Daily Precipitation (GUGBADP).
    This script can do a bias correction using the Intensity-Duraton-Frequency (IDF) curves 
    correction method
REQUIREMENT
    It required numpy, scipy and xarray module. So it will work on any machine environment
EXAMPLES
    python imerg_cpc_idfcurves.py
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
import os
import csv
import netCDF4
import numpy as np
from scipy.stats import pearsonr, mean_absolute_error

def calculate_relative_bias(observed, simulated):
    return (simulated - observed).sum() / observed.sum()

def calculate_metrics(observed, simulated):
    rmse = np.sqrt(((observed - simulated) ** 2).mean())
    mae = mean_absolute_error(observed, simulated)
    pod = ((observed > 0) & (simulated > 0)).sum() / (observed > 0).sum()
    far = ((observed == 0) & (simulated > 0)).sum() / (observed == 0).sum()
    csi = ((observed > 0) & (simulated > 0)).sum() / ((observed > 0) | (simulated > 0)).sum()
    return rmse, mae, pod, far, csi

def bias_correction(observed, simulated, corrected_output_path, factor_output_path, dekad_interval):
    # Calculate the relative bias
    relative_bias = calculate_relative_bias(observed, simulated)

    # Calculate the Pearson correlation coefficient and p-value
    pearson, _ = pearsonr(observed.flatten(), simulated.flatten())

    # Calculate the other metrics
    rmse, mae, pod, far, csi = calculate_metrics(observed, simulated)

    # Create the correcting factor by dividing the simulated data by the observed data
    correcting_factor = simulated / observed

    # Initialize the arrays for the corrected data and multiplying factors
    corrected = np.empty_like(observed)
    factors = np.empty_like(observed)

    # Loop through the time steps and apply the correction
    for i in range(observed.shape[0]):
        corrected[i] = observed[i] * correcting_factor[i]
        factors[i] = correcting_factor[i]

    # Create the NetCDF file for the corrected data
    with netCDF4.Dataset(corrected_output_path, 'w', format='NETCDF4') as out_file:
        # Create the dimensions
        out_file.createDimension('time', observed.shape[0])

            # Create the variables
        time = out_file.createVariable('time', 'f8', ('time',))
        corrected_data = out_file.createVariable('corrected_data', 'f4', ('time',))

        # Assign the values to the variables
        time[:] = np.arange(observed.shape[0])
        corrected_data[:] = corrected

    # Create the NetCDF file for the multiplying factors
    with netCDF4.Dataset(factor_output_path, 'w', format='NETCDF4') as out_file:
        # Create the dimensions
        out_file.createDimension('time', observed.shape[0])

        # Create the variables
        time = out_file.createVariable('time', 'f8', ('time',))
        factor = out_file.createVariable('factor', 'f4', ('time',))

        # Assign the values to the variables
        time[:] = np.arange(observed.shape[0])
        factor[:] = factors

    # Divide the multiplying factors by the number of days in each dekad
    num_dekads = observed.shape[0] // dekad_interval
    dekad_factors = np.empty((num_dekads, dekad_interval))
    for i in range(num_dekads):
        dekad_factors[i] = factors[i * dekad_interval : (i + 1) * dekad_interval] / np.arange(dekad_interval, 0, -1)

    # Output the dekad multiplying factors to separate NetCDF files
    for i in range(num_dekads):
        dekad_factor_output_path = f'{factor_output_path.rsplit(".", 1)[0]}_{i}.nc'
        with netCDF4.Dataset(dekad_factor_output_path, 'w', format='NETCDF4') as out_file:
            # Create the dimensions
            out_file.createDimension('time', dekad_interval)

            # Create the variables
            time = out_file.createVariable('time', 'f8', ('time',))
            factor = out_file.createVariable('factor', 'f4', ('time',))

            # Assign the values to the variables
            time[:] = np.arange(dekad_interval)
            factor[:] = dekad_factors[i]

    # Return the calculated metrics
    return relative_bias, pearson, rmse, mae, pod, far, csi

# Set the start and end years
start_year = 2001
end_year = 2022

# Initialize the list of metrics
metrics = []

# Loop through the years
for year in range(start_year, end_year + 1):
    # Load the observed and simulated data
    observed_path = f'imerg_{year}.nc'
        simulated_path = f'cpc_{year}.nc'
    with netCDF4.Dataset(observed_path, 'r') as in_file:
        observed = in_file['precipitation'][:]
    with netCDF4.Dataset(simulated_path, 'r') as in_file:
        simulated = in_file['precipitation'][:]

    # Set the output file paths for the corrected data and multiplying factors
    corrected_output_path = f'corrected_{year}.nc'
    factor_output_path = f'factors_{year}.nc'

    # Perform the bias correction and calculate the metrics
    relative_bias, pearson, rmse, mae, pod, far, csi = bias_correction(observed, simulated, corrected_output_path, factor_output_path, 10)

    # Add the metrics to the list
    metrics.append((year, relative_bias, pearson, rmse, mae, pod, far, csi))

# Output the metrics to a CSV file
with open('metrics.csv', 'w', newline='') as out_file:
    writer = csv.writer(out_file)
    writer.writerow(('year', 'relative_bias', 'pearson', 'rmse', 'mae', 'pod', 'far', 'csi'))
    writer.writerows(metrics)
