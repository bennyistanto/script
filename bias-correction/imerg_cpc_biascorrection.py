# -*- coding: utf-8 -*-
"""
NAME
    imerg_cpc_biascorrection.py
DESCRIPTION
    Bias correction using various methods:
    - Scale
    - Distribution
    - Delta
    - the Least Squares Composite differencing (LSC)
    - the Linear Scaling (LS) and quantile-mapping CDF matching approaches (LSCDF)
    - Replacement-based Cumulative Distribution Function (CDF) Mapping
REQUIREMENT
    It required os, calendar, numpy, xarray, pandas, scipy, and dask module. 
    So it will work on any machine environment
HOW-TO USE
    python imerg_cpc_biascorrection.py
NOTES
    Input data for this script will use GPM IMERG daily data compiled as 1 year 1 nc file
    and CPC Global Unified Gauge-Based Analysis of Daily Precipitation (GUGBADP).
    This script is designed to work with global daily IMERG and GUGBADP data, and compiled as
    1 year 1 nc file folowing GUGBDAP structure. To do this, you can use Climate Data Operator 
    (CDO) to manipulate the IMERG. Some CDO's module like mergetime and remapbil are useful to 
    merge daily data into annual data then re-grided following GUGBDAP spatial resolution. 
    After this steps are done, you can start the correction.
    Both variables in IMERG and GUGBDAP is written as "precipitationCal" and "precip"", 
    some adjustment are required: parsing filename, directory, variable name, etc.
WORKING DIRECTORY
    /input/imerg - put your IMERG data here
    /input/cpc - put your GUGBADP data here
    /output/{method}/corrected - location for corrected precipitation output
    /output/{method}/factors - location for corrected multiplying factors output
    /output/{method}/metrics - location for corrected statistical metrics output
DATA
    IMERG: 
    - Early: https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDE.06/
    - Late: https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDL.06/
    - Final: https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.06/
    GUGBADP: https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html
CONTACT
    Benny Istanto
    Climate Geographer
    GOST/DECAT/DEC Data Group, The World Bank
LICENSE
    This script is in the public domain, free from copyrights or restrictions.
VERSION
    $Id$
TODO
    xx
"""
import os
import calendar
import xarray as xr
import numpy as np
import pandas as pd

def bias_correction(imerg_ds, cpc_ds, method='rcdfm'):
    """
    Correct the bias in the IMERG data using the Replacement-based CDF Mapping method or other methods.

    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.
    - method (str): the method to use for correction, either 'rcdfm', 'scale', 'distribution' or other.

    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.
    """
    if method == 'scale':
        corrected_ds = scale(imerg_ds, cpc_ds)
    elif method == 'distribution':
        corrected_ds = distribution(imerg_ds, cpc_ds)
    elif method == 'delta':
        corrected_ds = delta(imerg_ds, cpc_ds)
    elif method == 'lsc':
        corrected_ds = lsc(imerg_ds, cpc_ds)
    elif method == 'lscdf':
        corrected_ds = lscdf(imerg_ds, cpc_ds)
    elif method == 'rcdfm':
        corrected_ds = rcdfm(imerg_ds, cpc_ds)
    # add more elif statement for other correction methods
        else:
        raise ValueError("Invalid method. Choose either 'scale', 'distribution', 'delta', 'lsc', 'lscdf', 'rcdfm'.")

    return corrected_ds

def scale(imerg_ds, cpc_ds, method='mean'):
    """
    Scale the IMERG data by a constant factor, determined by the ratio of the mean or median of the IMERG data
    to the mean or median of the CPC data.

    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.
    - method (str): the method to use to calculate the scaling factor, either 'mean' or 'median'

    Returns:
    - corrected_ds (xarray.Dataset): the scaled IMERG data.
    """
    def custom_scaling_factor(imerg_precip, cpc_precip, method='mean'):
	    """
	    Calculate the scaling factor for correcting the IMERG data using either the mean or median of 
	    the IMERG and CPC data.

	    Parameters:
	    - imerg_precip (xarray.DataArray): the IMERG precipitation data.
	    - cpc_precip (xarray.DataArray): the CPC precipitation data.
	    - method (str): the method to use to calculate the scaling factor, either 'mean' or 'median'

	    Returns:
	    - scaling_factor (xarray.DataArray): the scaling factor for correcting the IMERG data.
	    """
	    if method == 'mean':
	        imerg_mean = imerg_precip.mean()
	        cpc_mean = cpc_precip.mean()
	    elif method == 'median':
	        imerg_mean = xr.apply_ufunc(np.median, imerg_precip)
	        cpc_mean = xr.apply_ufunc(np.median, cpc_precip)
	    else:
	        raise ValueError("Invalid method. Choose either 'mean' or 'median'.")

	    scaling_factor = cpc_mean / imerg_mean

	    return scaling_factor

    scaling_factor = custom_scaling_factor(imerg_ds['precipitationCal'], cpc_ds['precip'], method)
    corrected_ds = xr.Dataset(data_vars={'precipitation': (('time', 'lat', 'lon'), 
    				xr.apply_ufunc(lambda x: x * scaling_factor, imerg_ds['precipitationCal'], 
    				dask='parallelized', output_dtypes=[float]))},
                              coords={'time': imerg_ds['time'],
                                      'lat': imerg_ds['lat'],
                                      'lon': imerg_ds['lon']})
    
    return corrected_ds

def distribution(imerg_ds, cpc_ds):
	"""
    Correct the bias in the IMERG data using the distribution-based method, specifically the probability 
    density function (PDF) matching method

    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.

    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.
    """
    from scipy import stats

    def pdf_matching(imerg_precip, cpc_precip):
        # Compute the probability density functions for the IMERG and CPC data
        imerg_pdf, _ = stats.gaussian_kde(imerg_precip)
        cpc_pdf, _ = stats.gaussian_kde(cpc_precip)

        # Map the IMERG data to the CPC PDF using linear interpolation
        corrected_precip = np.interp(imerg_precip, imerg_pdf, cpc_pdf)
        return corrected_precip

    # Get the precipitation data from the input datasets
    imerg_precip = imerg_ds['precipitationCal']
    cpc_precip = cpc_ds['precip']

    # Perform the correction
    corrected_precip = xr.apply_ufunc(pdf_matching, imerg_precip, cpc_precip,
                                      input_core_dims=[['time'], ['time']],
                                      output_core_dims=[['time']],
                                      dask='parallelized', output_dtypes=[float])

    # Create a new xarray.Dataset with the bias-corrected data
    corrected_ds = xr.Dataset(data_vars={'precipitation': (('time', 'lat', 'lon'), corrected_precip)},
                              coords={'time': imerg_ds['time'],
                                      'lat': imerg_ds['lat'],
                                      'lon': imerg_ds['lon']})
    
    return corrected_ds

def delta(imerg_ds, cpc_ds):
    """
    Correct the bias in the IMERG data using the delta method.

    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.

    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.
    """
    # Get the precipitation data from the input datasets
    imerg_precip = imerg_ds['precipitationCal']
    cpc_precip = cpc_ds['precip']

    # Compute the bias-corrected data using the delta method
    corrected_precip = imerg_precip + (cpc_precip - imerg_precip)

    # Define the bias correction function
    def delta_precip(imerg, cpc):
        return imerg + (cpc - imerg)
    
    # Apply the bias correction function using xarray's apply_ufunc
    corrected_precip = xr.apply_ufunc(delta_precip, imerg_precip, cpc_precip, dask='parallelized')

    # Create a new xarray.Dataset with the bias-corrected data
    corrected_ds = xr.Dataset(data_vars={'precipitation': (('time', 'lat', 'lon'), corrected_precip)},
                              coords={'time': imerg_ds['time'],
                                      'lat': imerg_ds['lat'],
                                      'lon': imerg_ds['lon']})
    
    return corrected_ds

def lsc(imerg_ds, cpc_ds):
    """
    Correct the bias in the IMERG data using the Least Squares Composite differencing (LSC) method.

    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.

    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.
    """
    from scipy.optimize import minimize
    
    # Get the precipitation data from the input datasets
    imerg_precip = imerg_ds['precipitationCal']
    cpc_precip = cpc_ds['precip']

    # Define the objective function that will be minimized
    def objective(c, imerg_precip, cpc_precip):
        """
        The objective function to be minimized in the LSCD method.

        Parameters:
        - c (float): the correction factor.
        - imerg_precip (xarray.DataArray): the IMERG precipitation data.
        - cpc_precip (xarray.DataArray): the CPC precipitation data.

        Returns:
        - objective_val (float): the value of the objective function for the given correction factor.
        """
        objective_val = np.sum((imerg_precip + c - cpc_precip)**2)
        return objective_val

    # Define the function to compute the LSCD correction
    def compute_lscd(imerg_precip, cpc_precip):
        """
        Compute the correction factor for the LSCD method.

        Parameters:
        - imerg_precip (xarray.DataArray): the IMERG precipitation data.
        - cpc_precip (xarray.DataArray): the CPC precipitation data.

        Returns:
        - correction (float): the correction factor.
        """
        # Initialize the correction factor
        c0 = 0.0

        # Perform the optimization
        res = minimize(objective, c0, args=(imerg_precip, cpc_precip))
        correction = res.x[0]
        return correction

    # Perform the correction
    correction = xr.apply_ufunc(compute_lscd, imerg_precip, cpc_precip,
                                input_core_dims=[['time'], ['time']],
                                output_core_dims=[[]],
                                dask='parallelized', output_dtypes=[float])

    # Apply the correction to the IMERG data
    corrected_precip = imerg_precip + correction

    # Create a new xarray.Dataset with the bias-corrected data
    corrected_ds = xr.Dataset(data_vars={'precipitation': (('time', 'lat', 'lon'), corrected_precip)},
                              coords={'time': imerg_ds['time'],
                                      'lat': imerg_ds['lat'],
                                      'lon': imerg_ds['lon']})
    
    return corrected_ds

def lscdf(imerg_ds, cpc_ds):
    """
    Correct the bias in the IMERG data using the Linear Scaling (LS) and quantile-mapping Cumulative 
    Distribution Function (CDF) matching approaches (LSCDF).

    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.

    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.
    """
    from scipy.stats import rankdata, linregress

    # Get the precipitation data from the input datasets
    imerg_precip = imerg_ds['precipitationCal']
    cpc_precip = cpc_ds['precip']

    # Compute the bias-corrected data using the LSCDF method
    corrected_precip = xr.apply_ufunc(lambda x, y: x + (linregress(rankdata(y), rankdata(x))[0] * (rankdata(x) - 0.5)),
                                      imerg_precip, cpc_precip, dask='parallelized', output_dtypes=[imerg_precip.dtype])

    # Create a new xarray.Dataset with the bias-corrected data
    corrected_ds = xr.Dataset(data_vars={'precipitation': (('time', 'lat', 'lon'), corrected_precip)},
                              coords={'time': imerg_ds['time'],
                                      'lat': imerg_ds['lat'],
                                      'lon': imerg_ds['lon']})
    
    return corrected_ds

def rcdfm(imerg_ds, cpc_ds):
    """
    Correct the bias in the IMERG data using the Replacement-based CDF Mapping method.

    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.

    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.
    """
    from scipy.stats import rankdata

    # Get the precipitation data from the input datasets
    imerg_precip = imerg_ds['precipitationCal']
    cpc_precip = cpc_ds['precip']
    
    # Get the empirical CDF of the CPC data
    cpc_cdf = xr.apply_ufunc(lambda x: rankdata(x)/x.size, cpc_precip, dask='parallelized')
    
    # Perform the correction
    corrected_precip = xr.where(imerg_precip!=0, imerg_precip.interp(precipitation=cpc_cdf), 0)
    
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
    imerg_precip = imerg_ds['precipitationCal']
    cpc_precip = cpc_ds['precip']

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
	"""
    The main process to calculate:
    - bias correction
    - save corrected data to netcdf
    - metrics
    - save metrics to csv
    - multiplying factors
    - save multiplying factors as netcdf

    Returns:
    - Output available in folder output/{method}/ corrected, metrics and factors
    """
    from dask import delayed
    
    # Get the correction method used
    method = bias_correction.__defaults__[0]

    # Define the appropriate input and output directory paths
    input_dir = f'input'
    imerg_path = f'{input_dir}/imerg'
    cpc_path = f'{input_dir}/cpc'
    output_dir = f'output'
    method_dir = f'output/{method}'
    corrected_path = f'{method_dir}/corrected'
    factors_path = f'{method_dir}/factors'
    metrics_path = f'{method_dir}/metrics'

    # Create the output directories if they don't already exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(method_dir, exist_ok=True)
    os.makedirs(corrected_path, exist_ok=True)
    os.makedirs(factors_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    # Load the IMERG data
    imerg_ds = xr.open_mfdataset('{imerg_path}/*.nc')

    # Load the CPC data
    cpc_ds = xr.open_mfdataset('{cpc_path}/*.nc')

    # Initialize an empty dataframe to store the metric values
    metrics = pd.DataFrame(columns=['relative_bias', 'pearson', 'rmse', 'mae', 'pod', 'far', 'csi'])

    # Use dask's delayed function to schedule the computation for each year in parallel
    corrected_ds_list = []
    for year in range(2001, 2023):
        # Get the data for this year
        imerg_year_ds = imerg_ds.sel(time=imerg_ds['time.year'] == year)
        cpc_year_ds = cpc_ds.sel(time=cpc_ds['time.year'] == year)

        # Schedule the calculation of the corrected data for this year
        corrected_ds = delayed(bias_correction)(imerg_year_ds, cpc_year_ds)
        corrected_ds_list.append(corrected_ds)

    # Compute the corrected data for all years in parallel
    corrected_ds_list = dask.compute(*corrected_ds_list)

    # Save the corrected data to a NetCDF file
    for year, corrected_ds in zip(range(2001, 2023), corrected_ds_list):
        corrected_ds.to_netcdf(f'{corrected_path}/corrected_{year}.nc')

    # Calculate the metrics for the corrected data
    for corrected_ds in corrected_ds_list:
        year_metrics = calculate_metrics(corrected_ds, cpc_year_ds)

        # Add the metric values for this year to the dataframe
        metrics = metrics.append(year_metrics, ignore_index=True)

    # Output the metric values to a CSV file
    metrics.to_csv(f'{metrics_path}/metrics.csv', index=False)

    # Create the multiplying factor files
    dekad_factors = create_multiplying_factors(corrected_ds)
    for i, factor in enumerate(dekad_factors):
        factor.to_netcdf(f'{factors_path}/multiplying_factor_{i+1}.nc')

if __name__ == '__main__':
    main()
