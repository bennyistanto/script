# -*- coding: utf-8 -*-
"""
NAME
    imerg_cpc_biascorr.py
    Bias correction using LSCDF
DESCRIPTION
    Input data for this script will use GPM IMERG daily data compiled as 1 year 1 nc file
    and CPC Global Unified Gauge-Based Analysis of Daily Precipitation (GUGBADP).
    This script can do a bias correction using the scaling, distribution-based and 
    delta method correction
REQUIREMENT
    It required numpy, scipy and xarray module. So it will work on any machine environment
EXAMPLES
    python imerg_cpc_biascorr.py
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
from scipy.stats import pearsonr, rankdata
import calendar

def bias_correction(imerg, cpc):
    """
    Perform bias correction on the given IMERG and CPC data.
    
    Parameters:
    - imerg: xarray DataArray of IMERG data
    - cpc: xarray DataArray of CPC data
    
    Returns:
    - corrected: xarray DataArray of corrected IMERG data
    - stats: dictionary containing various statistics about the correction
    """
    # Scale the IMERG data to match the CPC data
    scale_factor = cpc.mean() / imerg.mean()
    corrected = imerg * scale_factor
    
    # Calculate relative bias
    relative_bias = (cpc.mean() - corrected.mean()) / cpc.mean()
    
    # Calculate Pearson's correlation coefficient
    pearson, _ = pearsonr(cpc.values.flatten(), corrected.values.flatten())
    
    # Calculate root mean squared error
    rmse = np.sqrt(((cpc - corrected) ** 2).mean())
    
    # Calculate mean absolute error
    mae = np.abs(cpc - corrected).mean()
    
    # Calculate probability of detection
    pod = np.sum((cpc > 0) & (corrected > 0)) / np.sum(cpc > 0)
    
    # Calculate false alarm ratio
    far = np.sum((cpc == 0) & (corrected > 0)) / np.sum(cpc == 0)
    
    # Calculate critical success index
    csi = np.sum((cpc > 0) & (corrected > 0)) / (np.sum((cpc > 0) & (corrected > 0)) + np.sum((cpc == 0) & (corrected > 0)) + np.sum((cpc > 0) & (corrected == 0)))
    
    # Return the corrected data and the statistics
    stats = {
        'relative_bias': relative_bias,
        'pearson': pearson,
        'rmse': rmse,
        'mae': mae,
        'pod': pod,
        'far': far,
        'csi': csi
    }
    return corrected, stats

def distribution_based_correction(imerg, cpc):
    """
    Perform distribution-based bias correction on the given IMERG and CPC data.
    
    Parameters:
    - imerg: xarray DataArray of IMERG data
    - cpc: xarray DataArray of CPC data
    
    Returns:
    - corrected: xarray DataArray of corrected IMERG data
    - stats: dictionary containing various statistics about the correction
    """
    # Calculate the ranks of the data
    imerg_ranks = rankdata(imerg.values.flatten())
    cpc_ranks = rankdata(cpc.values.flatten())
    
    # Calculate the quantiles of the data
    imerg_quantiles = imerg_ranks / imerg_ranks.size
    cpc_quantiles = cpc_ranks / cpc_ranks.size
    
    # Interpolate the IMERG data to match the CPC data
    corrected = xr.DataArray(np.interp(imerg_quantiles, cpc_quantiles, cpc.values.flatten()).reshape(imerg.shape), coords=imerg.coords, dims=imerg.dims)
    
    # Calculate relative bias
    relative_bias = (cpc.mean() - corrected.mean()) / cpc.mean()
    
    # Calculate Pearson's correlation coefficient
    pearson, _ = pearsonr(cpc.values.flatten(), corrected.values.flatten())
    
    # Calculate root mean squared error
    rmse = np.sqrt(((cpc - corrected) ** 2).mean())
    
    # Calculate mean absolute error
    mae = np.abs(cpc - corrected).mean()
    
    # Calculate probability of detection
    pod = np.sum((cpc > 0) & (corrected > 0)) / np.sum(cpc > 0)
    
    # Calculate false alarm ratio
    far = np.sum((cpc == 0) & (corrected > 0)) / np.sum(cpc == 0)
    
    # Calculate critical success index
    csi = np.sum((cpc > 0) & (corrected > 0)) / (np.sum((cpc > 0) & (corrected > 0)) + np.sum((cpc == 0) & (corrected > 0)) + np.sum((cpc > 0) & (corrected == 0)))
    
    # Return the corrected data and the statistics
    stats = {
        'relative_bias': relative_bias,
        'pearson': pearson,
        'rmse': rmse,
        'mae': mae,
        'pod': pod,
        'far': far,
        'csi': csi
    }
    return corrected, stats

def delta_method_correction(imerg, cpc):
    """
    Perform delta method bias correction on the given IMERG and CPC data.
    
    Parameters:
    - imerg: xarray DataArray of IMERG data
    - cpc: xarray DataArray of CPC data
    
    Returns:
    - corrected: xarray DataArray of corrected IMERG data
    - stats: dictionary containing various statistics about the correction
    """
    # Calculate the delta factor
    delta = cpc.mean() - imerg.mean()
    
    # Correct the IMERG data using the delta factor
    corrected = imerg + delta
    
    # Calculate relative bias
    relative_bias = (cpc.mean() - corrected.mean()) / cpc.mean()
    
    # Calculate Pearson's correlation coefficient
    pearson, _ = pearsonr(cpc.values.flatten(), corrected.values.flatten())
    
    # Calculate root mean squared error
    rmse = np.sqrt(((cpc - corrected) ** 2).mean())
    
    # Calculate mean absolute error
    mae = np.abs(cpc - corrected).mean()
    
    # Calculate probability of detection
    pod = np.sum((cpc > 0) & (corrected > 0)) / np.sum(cpc > 0)
    
    # Calculate false alarm ratio
    far = np.sum((cpc == 0) & (corrected > 0)) / np.sum(cpc == 0)
    
    # Calculate critical success index
    csi = np.sum((cpc > 0) & (corrected > 0)) / (np.sum((cpc > 0) & (corrected > 0)) + np.sum((cpc == 0) & (corrected > 0)) + np.sum((cpc > 0) & (corrected == 0)))
    
    # Return the corrected data and the statistics
        stats = {
        'relative_bias': relative_bias,
        'pearson': pearson,
        'rmse': rmse,
        'mae': mae,
        'pod': pod,
        'far': far,
        'csi': csi
    }
    return corrected, stats

# Loop over all of the years of data
for year in range(2001, 2022):
    # Open the IMERG and CPC data for the current year
    imerg = xr.open_dataset(f'imerg_{year}.nc')['precipitation']
    cpc = xr.open_dataset(f'cpc_{year}.nc')['precipitation']
    
    # Check if the current year is a leap year
    leap_year = calendar.isleap(year)
    
    # Calculate the multiplying factors for every 10-day period (dekad) in the year
    dekad_factors = {}
    for dekad in range(1, 37):
        # Calculate the start and end dates of the dekad
        start_day = (dekad - 1) * 10 + 1
        end_day = dekad * 10
        # Check if the dekad includes the end of February in a leap year
        if leap_year and dekad == 4:
            end_day += 1
        # Check if the dekad includes the end of December
        elif dekad == 36:
            end_day += 1
        
        # Perform the bias correction using each method
        scaled, stats = bias_correction(imerg.sel(time=slice(f'{year}-01-{start_day:02d}', f'{year}-01-{end_day:02d}')), cpc.sel(time=slice(f'{year}-01-{start_day:02d}', f'{year}-01-{end_day:02d}')))
        distributed, stats = distribution_based_correction(imerg.sel(time=slice(f'{year}-01-{start_day:02d}', f'{year}-01-{end_day:02d}')), cpc.sel(time=slice(f'{year}-01-{start_day:02d}', f'{year}-01-{end_day:02d}')))
        delta, stats = delta_method_correction(imerg.sel(time=slice(f'{year}-01-{start_day:02d}', f'{year}-01-{end_day:02d}')), cpc.sel(time=slice(f'{year}-01-{start_day:02d}', f'{year}-01-{end_day:02d}')))
        
        # Calculate the multiplying factors
        scaled_factor = scaled / imerg.sel(time=slice(f'{year}-01-{start_day:02d}', f'{year}-01-{end_day:02d}'))
        distributed_factor = distributed / imerg.sel(time=slice(f'{year}-01-{start_day:02d}', f'{year}-01-{end_day:02d}'))
        delta_factor = delta / imerg.sel(time=slice(f'{year}-01-{start_day:02d}', f'{year}-01-{end_day:02d}'))
        
        # Divide the multiplying factors by the number of days in the dekad
        num_days = end_day - start_day + 1
        scaled_factor /= num_days
        distributed_factor /= num_days
        delta_factor /= num_days
        
        # Add the multiplying factors to the dictionary
        dekad_factors[dekad] = {
            'scaled': scaled_factor,
            'distributed': distributed_factor,
            'delta': delta_factor
        }
    
    # Correct the IMERG data for the entire year using each method
    scaled_corrected, _ = bias_correction(imerg, cpc)
    distributed_corrected, _ = distribution_based_correction(imerg, cpc)
    delta_corrected, _ = delta_method_correction(imerg, cpc)
    
    # Save the corrected data and multiplying factors for each method
    scaled_corrected.to_netcdf(f'imerg_scaled_{year}.nc')
    xr.Dataset({'factor': xr.concat(list(dekad_factors[dekad]['scaled'] for dekad in dekad_factors), 'dekad')}).to_netcdf(f'imerg_scaled_factors_{year}.nc')
    distributed_corrected.to_netcdf(f'imerg_distributed_{year}.nc')
    xr.Dataset({'factor': xr.concat(list(dekad_factors[dekad]['distributed'] for dekad in dekad_factors), 'dekad')}).to_netcdf(f'imerg_distributed_factors_{year}.nc')
    delta_corrected.to_netcdf(f'imerg_delta_{year}.nc')
    xr.Dataset({'factor': xr.concat(list(dekad_factors[dekad]['delta'] for dekad in dekad_factors), 'dekad')}).to_netcdf(f'imerg_delta_factors_{year}.nc')
    
    # Close the IMERG and CPC data for the current year
    imerg.close()
    cpc.close()

# Create an empty DataFrame to store the metrics for all years
metrics = pd.DataFrame(columns=['year', 'method', 'relative_bias', 'pearson', 'rmse', 'mae', 'pod', 'far', 'csi'])

# Loop over all of the years of data
for year in range(2001, 2022):
    # Open the IMERG and CPC data for the current year
    imerg = xr.open_dataset(f'imerg_{year}.nc')['precipitation']
    cpc = xr.open_dataset(f'cpc_{year}.nc')['precipitation']
    
    # Perform the bias correction using each method
    _, scaled_stats = bias_correction(imerg, cpc)
    _, distributed_stats = distribution_based_correction(imerg, cpc)
    _, delta_stats = delta_method_correction(imerg, cpc)
    
    # Add the metrics for each method to the DataFrame
    metrics = metrics.append([
        {'year': year, 'method': 'scaled', **scaled_stats},
        {'year': year, 'method': 'distributed', **distributed_stats},
        {'year': year, 'method': 'delta', **delta_stats}
    ])
    
    # Close the IMERG and CPC data for the current year
    imerg.close()
    cpc.close()

# Save the DataFrame to a CSV file
metrics.to_csv('metrics.csv', index=False)