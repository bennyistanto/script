# -*- coding: utf-8 -*-
"""
NAME
    imerg_cpc_lscdf02.py
    Bias correction using LSCDF
DESCRIPTION
    Input data for this script will use GPM IMERG daily data compiled as 1 year 1 nc file
    and CPC Global Unified Gauge-Based Analysis of Daily Precipitation (GUGBADP).
    This script can do a bias correction using the Least Squares Composite Differencing (LSCDF)
REQUIREMENT
    It required numpy, xarray and pandas module. So it will work on any machine environment
EXAMPLES
    python imerg_cpc_lscdf02.py
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
import pandas as pd
import calendar
import datetime

def lscdf(reference, target):
    """
    Perform bias correction using the Least Squares Composite Differencing (LSCDF) method.
    """
    # Subtract the mean from both the reference and target datasets
    reference_mean = reference.mean(dim='time')
    target_mean = target.mean(dim='time')
    reference_demeaned = reference - reference_mean
    target_demeaned = target - target_mean
    
    # Calculate the slope and intercept of the linear regression of the demeaned datasets
    slope, intercept = np.polyfit(reference_demeaned, target_demeaned, 1)
    
    # Correct the target dataset using the slope and intercept
    corrected = (target - intercept) / slope
    
    return corrected

# Loop over the years in the data
for year in range(2001, 2022):
    # Load the IMERG and CPC data for the current year
    imerg = xr.open_dataset(f'imerg_{year}.nc')
    cpc = xr.open_dataset(f'cpc_{year}.nc')
    
    # Extract the daily precipitation data from the datasets
    imerg_precip = imerg['precipitation']
    cpc_precip = cpc['precipitation']
    
    # Bias correct the IMERG data using the LSCDF method
    corrected_imerg = lscdf(cpc_precip, imerg_precip)
    
    # Save the corrected data to a new NetCDF file
    corrected_imerg.to_netcdf(f'corrected_imerg_{year}.nc')
    
    # Calculate the metrics for the corrected data
    relative_bias = ((corrected_imerg - cpc_precip) / cpc_precip).mean(dim='time')
    pearson = corrected_imerg.pearson(cpc_precip, dim='time')
    rmse = (((corrected_imerg - cpc_precip) ** 2).mean(dim='time')) ** 0.5
    mae = (abs(corrected_imerg - cpc_precip)).mean(dim='time')

    # Calculate the probability of detection (POD), false alarm rate (FAR), and critical success index (CSI)
    # using the corrected data and the reference data
    # (assuming that the reference data is considered "observed" and the corrected data is considered "forecast")
    threshold = 0.1 # Set the threshold for considering a forecast "correct"

    # Count the number of times the forecast is above the threshold and the observed is above the threshold
    hits = np.logical_and(corrected_imerg > threshold, cpc_precip > threshold).sum(dim='time')
    # Count the number of times the forecast is above the threshold and the observed is below the threshold
    false_alarms = np.logical_and(corrected_imerg > threshold, cpc_precip < threshold).sum(dim='time')
    # Count the number of times the forecast is below the threshold and the observed is above the threshold
    misses = np.logical_and(corrected_imerg < threshold, cpc_precip > threshold).sum(dim='time')
    
    # Calculate the probability of detection (POD)
    pod = hits / (hits + misses)
    
    # Calculate the false alarm rate (FAR)
    far = false_alarms / (hits + false_alarms)
    
    # Calculate the critical success index (CSI)
    csi = hits / (hits + misses + false_alarms)

    # Append the metrics for the current year to a list
    metrics = [relative_bias, pearson, rmse, mae, pod, far, csi]
    all_metrics.append(metrics)

    # Create the multiplying factor file in netCDF format
    # First, determine the number of days in each dekad
    days_per_dekad = []
    for month in range(1, 13):
        if month in [1, 3, 5, 7, 8, 10, 12]:
            days_per_dekad.append(10)
        elif month == 2:
            if calendar.isleap(year):
                days_per_dekad.append(10)
            else:
                days_per_dekad.append(9)
        else:
            days_per_dekad.append(11)
    
    # Create an array of dekad dates
    dekad_dates = []
    for month in range(1, 13):
        for dekad in range(1, 4):
            dekad_dates.append(datetime.datetime(year, month, dekad*10))
    
    # Calculate the mean correction factor for each dekad
    correction_factors = []
    for i in range(0, len(days_per_dekad)):
        correction_factors.append(corrected_imerg[i*10:(i+1)*10].mean() / imerg_precip[i*10:(i+1)*10].mean())
    
    # Divide the correction factors by the number of days in each dekad to get the daily correction factors
    daily_correction_factors = [factor / days for factor, days in zip(correction_factors, days_per_dekad)]
    
    # Create a NetCDF file for each dekad
    for i in range(0, len(dekad_dates)):
        dekad_date = dekad_dates[i]
        dekad_correction_factor = daily_correction_factors[i]
        
        # Create a dataset containing the correction factor for this dekad
        dekad_dataset = xr.Dataset({'correction_factor': (['time'], [dekad_correction_factor])},
                                   coords={'time': (['time'], [dekad_date])})
    
        # Save the dataset to a NetCDF file
        dekad_dataset.to_netcdf(f'multiplying_factor_{dekad_date:%Y%m%d}.nc')

    # Save the metrics for all years to a CSV file
    columns = ['relative_bias', 'pearson', 'rmse', 'mae', 'pod', 'far', 'csi']
    metrics_df = pd.DataFrame(all_metrics, columns=columns)
    metrics_df.to_csv('metrics.csv')