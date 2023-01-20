# -*- coding: utf-8 -*-
"""
NAME
    imerg_cpc_biascorrection.py
DESCRIPTION
    Bias correction using various methods:
    i. Scale, ii. Distribution, iii. Delta, iv. the Least Squares Composite differencing (LSC)
    v. the Linear Scaling (LS) and quantile-mapping CDF matching approaches (LSCDF),
    vi.  Replacement-based Cumulative Distribution Function (RCDF) Mapping,
    vii. Multi Linear ERegression, viii. Artificial Neural Network, ix. Kalman filtering,
    x. Bias Correction and Spatial Disaggregation (BCSD), 
    xi. Bias Correction and Spatially  Disaggregated Mapping (BCSDM), xii. Quantile-quantile mapping
    xiii. Empirical Quantile Mapping (eQM), xiv. Adjusted Quantile Mapping (aQM)
    xv. 
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
        corrected_ds = scale(imerg_ds, cpc_ds, method='mean')
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
    elif method == 'mlr':
        corrected_ds = mlr(imerg_ds, cpc_ds)
    elif method == 'ann':
        corrected_ds = ann(imerg_ds, cpc_ds)
    elif method == 'kalman':
        corrected_ds = kalman(imerg_ds, cpc_ds)
    elif method == 'bcsd':
        corrected_ds = bcsd(imerg_ds, cpc_ds, method='linear')
    elif method == 'bcsdm':
        corrected_ds = bcsdm(imerg_ds, cpc_ds, method='linear')
    elif method == 'qq_mapping':
        corrected_ds = qq_mapping(imerg_ds, cpc_ds)
    elif method == 'eQM':
        corrected_ds = eQM(cpc_ds, imerg_ds)
    elif method == 'aQM':
        corrected_ds = aQM(cpc_ds, imerg_ds, alpha=0.5)
    elif method == 'gQM':
        corrected_ds = gQM(cpc_ds, imerg_ds)
    elif method == 'gpdQM':
        corrected_ds = gpdQM(cpc_ds, imerg_ds)
    # add more elif statement for other correction methods
        
        else:
        raise ValueError("Invalid method. Choose either 'scale', 'distribution', 'delta', 'lsc', 'lscdf', \
            'rcdfm', 'mlr', 'ann', 'kalman', 'bcsd', 'bcsdm', 'qq_mapping', 'eQM', 'aQM', 'gQM', 'gpdQM'.")

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
	    Calculate the scaling factor for correcting the IMERG data using either the mean or median of the IMERG and CPC data.

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
    Correct the bias in the IMERG data using the distribution-based method, specifically the probability density function (PDF) matching method

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
    Correct the bias in the IMERG data using the Linear Scaling (LS) and quantile-mapping Cumulative Distribution Function (CDF) matching approaches (LSCDF).

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

def mlr(imerg_ds, cpc_ds):
    """
    Correct the bias in the IMERG data using multiple linear regression.
    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.
    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.

    This function uses the LinearRegression from scikit-learn library, it flattens 
    the data to 2D arrays, then gets the lat and lon as additional predictors and 
    stack it with imerg_precip, then trains the multiple linear regression model using 
    the stacked predictors as input and cpc_precip as the target, and then applies 
    the trained model to predict the corrected precipitation. The corrected precipitation 
    is then reshaped back to the original shape before returning the corrected dataset.
    """
    from sklearn.linear_model import LinearRegression

    # Get the precipitation data from the input datasets
    imerg_precip = imerg_ds['precipitationCal'].values
    cpc_precip = cpc_ds['precip'].values

    # Flatten the data to 2D array
    imerg_precip = imerg_precip.reshape(-1,1)
    cpc_precip = cpc_precip.reshape(-1,1)

    # Get the lats and lons as additional predictors
    lats = imerg_ds['lat'].values
    lons = imerg_ds['lon'].values

    # Stack the predictors
    predictors = np.column_stack((imerg_precip, lats, lons))

    # Train the multiple linear regression model
    mlr = LinearRegression()
    mlr.fit(predictors, cpc_precip)

    # Use the trained model to predict the corrected precipitation
    corrected_precip = mlr.predict(predictors)

    # Define the bias correction function
    def mlr_precip(imerg, cpc, lats, lons):
        predictors = np.column_stack((imerg, lats, lons))
        corrected_precip = mlr.predict(predictors)
        
        return corrected_precip
    
    # Apply the bias correction function using xarray's apply_ufunc
    corrected_precip = xr.apply_ufunc(mlr_precip, imerg_precip, cpc_precip, lats, lons, dask='parallelized')

    # Create a new xarray.Dataset with the bias-corrected data
    corrected_ds = xr.Dataset(data_vars={'precipitation': (('time', 'lat', 'lon'), corrected_precip)},
                              coords={'time': imerg_ds['time'],
                                      'lat': imerg_ds['lat'],
                                      'lon': imerg_ds['lon']})
    
    return corrected_ds

def ann(imerg_ds, cpc_ds):
    """
    Correct the bias in the IMERG data using an artificial neural network.
    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.
    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.

    This function uses the MLPRegressor from scikit-learn library as an example of ANN, 
    it flattens the data to 2D arrays, trains the ANN model using imerg_precip as input 
    and cpc_precip as the target, then applies the trained model to predict the corrected 
    precipitation. The corrected precipitation is then reshaped back to the original shape 
    before returning the corrected dataset.
    """
    from sklearn.neural_network import MLPRegressor

    # Get the precipitation data from the input datasets
    imerg_precip = imerg_ds['precipitationCal'].values
    cpc_precip = cpc_ds['precip'].values

    # Flatten the data to 2D array
    imerg_precip = imerg_precip.reshape(-1,1)
    cpc_precip = cpc_precip.reshape(-1,1)

    # Train the ANN model
    ann = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=1000)
    ann.fit(imerg_precip, cpc_precip)

    # Use the trained ANN model to predict the corrected precipitation
    corrected_precip = ann.predict(imerg_precip)

    # Define the bias correction function
    def ann_precip(imerg):
        
        return ann.predict(imerg)
    
    # Apply the bias correction function using xarray's apply_ufunc
    corrected_precip = xr.apply_ufunc(ann_precip, imerg_precip, dask='parallelized')

    # Create a new xarray.Dataset with the bias-corrected data
    corrected_ds = xr.Dataset(data_vars={'precipitation': (('time', 'lat', 'lon'), corrected_precip)},
                              coords={'time': imerg_ds['time'],
                                      'lat': imerg_ds['lat'],
                                      'lon': imerg_ds['lon']})
    
    return corrected_ds

def kalman(imerg_ds, cpc_ds):
    """
    Correct the bias in the IMERG data using Kalman filtering.
    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.
    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.

    This function uses the KalmanFilter from pykalman library, it flattens the data to 
    2D arrays, uses the Expectation-Maximization (EM) algorithm to estimate the parameters 
    of the Kalman filter and use it to predict the corrected precipitation, then applies 
    the trained model to predict the corrected precipitation. The corrected precipitation 
    is then reshaped back to the original shape before returning the corrected dataset.
    """
    from pykalman import KalmanFilter

    # Get the precipitation data from the input datasets
    imerg_precip = imerg_ds['precipitationCal'].values
    cpc_precip = cpc_ds['precip'].values

    # Flatten the data to 2D array
    imerg_precip = imerg_precip.reshape(-1,1)
    cpc_precip = cpc_precip.reshape(-1,1)

    # Define the Kalman filter
    kf = KalmanFilter(initial_state_mean=imerg_precip[0], n_dim_obs=1)
    kf = kf.em(cpc_precip, n_iter=5)

    # Use the Kalman filter to predict the corrected precipitation
    corrected_precip, _ = kf.filter(cpc_precip)

    # Define the bias correction function
    def kalman_precip(imerg, cpc):
        kf = KalmanFilter(initial_state_mean=imerg[0], n_dim_obs=1)
        kf = kf.em(cpc, n_iter=5)
        corrected_precip, _ = kf.filter(cpc)
        
        return corrected_precip
    
    # Apply the bias correction function using xarray's apply_ufunc
    corrected_precip = xr.apply_ufunc(kalman_precip, imerg_precip, cpc_precip, dask='parallelized')

    # Create a new xarray.Dataset with the bias-corrected data
    corrected_ds = xr.Dataset(data_vars={'precipitation': (('time', 'lat', 'lon'), corrected_precip)},
                              coords={'time': imerg_ds['time'],
                                      'lat': imerg_ds['lat'],
                                      'lon': imerg_ds['lon']})
    
    return corrected_ds

def bcsd(imerg_ds, cpc_ds, method='linear'):
    """
    Correct the bias in the IMERG data using Bias Correction and Spatial Disaggregation (BCSD) method.
    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.
    - method (str): method of interpolation, by default is linear
    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.

    This function uses the griddata from scipy.interpolate library, it first gets the lat 
    and lon coordinates of the CPC data, then it uses the griddata function to interpolate 
    the CPC data to the grid of the IMERG data. Interpolation method is set to linear by 
    default, but you can set it to another method if you want.
    """
    from scipy.interpolate import griddata

    # Get the precipitation data from the input datasets
    imerg_precip = imerg_ds['precipitationCal'].values
    cpc_precip = cpc_ds['precip'].values

    # Get the coordinates of the CPC data
    lats = cpc_ds['lat'].values
    lons = cpc_ds['lon'].values
    coords = np.array(list(zip(lats, lons)))
    
    # Interpolate the CPC data to the grid of the IMERG data
    corrected_precip = griddata(coords, cpc_precip, (imerg_ds.lat, imerg_ds.lon), method=method)

    # Define the bias correction function
    def bcsd_precip(imerg, cpc, lats, lons):
        coords = np.array(list(zip(lats, lons)))
        corrected_precip = griddata(coords, cpc, (imerg.lat, imerg.lon), method=method)
        
        return corrected_precip
    
    # Apply the bias correction function using xarray's apply_ufunc
    corrected_precip = xr.apply_ufunc(bcsd_precip, imerg_precip, cpc_precip, lats, lons, dask='parallelized')

    # Create a new xarray.Dataset with the bias-corrected data
    corrected_ds = xr.Dataset(data_vars={'precipitation': (('time', 'lat', 'lon'), corrected_precip)},
                              coords={'time': imerg_ds['time'],
                                      'lat': imerg_ds['lat'],
                                      'lon': imerg_ds['lon']})
    
    return corrected_ds

def bcsdm(imerg_ds, cpc_ds, method='linear'):
    """
    Correct the bias in the IMERG data using Bias Correction and Spatially Disaggregated Mapping (BCSDM) method.
    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.
    - method (str): method of interpolation, by default is linear
    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.
    """
    from scipy.interpolate import griddata

    # Get the precipitation data from the input datasets
    imerg_precip = imerg_ds['precipitationCal'].values
    cpc_precip = cpc_ds['precip'].values

    # Flatten the data to 2D array
    imerg_precip = imerg_precip.reshape(-1,1)
    cpc_precip = cpc_precip.reshape(-1,1)

    # Get the coordinates of the CPC data
    lats = cpc_ds['lat'].values
    lons = cpc_ds['lon'].values
    coords = np.array(list(zip(lats, lons)))
    
    # Interpolate the CPC data to the grid of the IMERG data
    corrected_precip = griddata(coords, cpc_precip, (imerg_ds.lat, imerg_ds.lon), method=method)

    # Define the bias correction function
    def bcsdm_precip(imerg, cpc, lats, lons):
        coords = np.array(list(zip(lats, lons)))
        corrected_precip = griddata(coords, cpc, (imerg.lat, imerg.lon), method=method)
        
        return corrected_precip
    
    # Apply the bias correction function using xarray's apply_ufunc
    corrected_precip = xr.apply_ufunc(bcsdm_precip, imerg_precip, cpc_precip, lats, lons, dask='parallelized')

    # Create a new xarray.Dataset with the bias-corrected data
    corrected_ds = xr.Dataset(data_vars={'precipitation': (('time', 'lat', 'lon'), corrected_precip)},
                              coords={'time': imerg_ds['time'],
                                      'lat': imerg_ds['lat'],
                                      'lon': imerg_ds['lon']})
    
    return corrected_ds

def qq_mapping(imerg_ds, cpc_ds):
    """
    Correct the bias in the IMERG data using quantile-quantile mapping (QQ-mapping) method.
    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.
    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.

    This function uses the rankdata from scipy.stats library, it flattens the data to 
    1D arrays, then computes the cumulative distribution function (CDF) of the IMERG data 
    and CPC data. Then it uses the np.interp to interpolate the CPC CDF to the IMERG CDF, 
    and this will give the correct precipitation values. The corrected precipitation is 
    then reshaped back to the original shape before returning the corrected dataset.
    """
    from scipy.stats import rankdata

    # Get the precipitation data from the input datasets
    imerg_precip = imerg_ds['precipitationCal'].values
    cpc_precip = cpc_ds['precip'].values

    # Flatten the data to 1D array
    imerg_precip = imerg_precip.flatten()
    cpc_precip = cpc_precip.flatten()

    # Compute the cumulative distribution function (CDF) of the IMERG data
    imerg_cdf = rankdata(imerg_precip) / len(imerg_precip)

    # Compute the CDF of the CPC data
    cpc_cdf = rankdata(cpc_precip) / len(cpc_precip)

    # Interpolate the CPC CDF to the IMERG CDF
    corrected_precip = np.interp(imerg_cdf, cpc_cdf, cpc_precip)

    # Define the bias correction function
    def qq_mapping_precip(imerg, cpc):
        imerg_cdf = rankdata(imerg) / len(imerg)
        cpc_cdf = rankdata(cpc) / len(cpc)
        corrected_precip = np.interp(imerg_cdf, cpc_cdf, cpc)
        
        return corrected_precip
    
    # Apply the bias correction function using xarray's apply_ufunc
    corrected_precip = xr.apply_ufunc(qq_mapping_precip, imerg_precip, cpc_precip, dask='parallelized')

    # Create a new xarray.Dataset with the bias-corrected data
    corrected_ds = xr.Dataset(data_vars={'precipitation': (('time', 'lat', 'lon'), corrected_precip)},
                              coords={'time': imerg_ds['time'],
                                      'lat': imerg_ds['lat'],
                                      'lon': imerg_ds['lon']})
    
    return corrected_ds

def eQM_(cpc_ds, imerg_ds):
    """
    Correct the bias in the simulated data using Empirical Quantile Mapping (eQM) method.
    Parameters:
    - cpc_ds (xarray.Dataset): the observed data to use as a reference for correction.
    - imerg_ds (xarray.Dataset): the simulated data to be corrected.
    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected simulated data.
    
    This function uses the scipy.stats.percentileofscore() function to compute the percentiles 
    of the observed and simulated datasets and use numpy.interp() to interpolate the percentiles 
    of the observed data to the percentiles of the simulated data, this will give the correct 
    precipitation values. The corrected precipitation is then reshaped back to the original 
    shape before returning the corrected dataset.
    """
    from scipy.stats import percentileofscore

    # Get the precipitation data from the input datasets
    cpc_precip = cpc_ds['precip'].values
    imerg_precip = imerg_ds['precipitationCal'].values

    # Compute the percentiles of the observed and simulated data
    cpc_percentiles = np.array([percentileofscore(cpc_precip, val, kind='rank') for val in cpc_precip])
    imerg_percentiles = np.array([percentileofscore(imerg_precip, val, kind='rank') for val in imerg_precip])
    
    # Define the bias correction function
    def eQM_precip(cpc, imerg):
        cpc_percentiles = np.array([percentileofscore(cpc_precip, val, kind='rank') for val in cpc_precip])
        imerg_percentiles = np.array([percentileofscore(imerg_precip, val, kind='rank') for val in imerg_precip])
        corrected = np.interp(imerg_percentiles, cpc_percentiles, cpc_precip)
        
        return corrected

    corrected_precip = xr.apply_ufunc(eQM_precip, cpc_precip, imerg_precip, dask='parallelized')
    
    # Create a new xarray.Dataset with the bias-corrected data
    corrected_ds = xr.Dataset(data_vars={'precipitation': (('time', 'lat', 'lon'), corrected_precip)},
                              coords={'time': imerg_ds['time'],
                                      'lat': imerg_ds['lat'],
                                      'lon': imerg_ds['lon']})
    
    return corrected_ds

def aQM(cpc_ds, imerg_ds, alpha=0.5):
    """
    Correct the bias in the IMERG data using Adjusted Quantile Mapping (AQM) method.
    Parameters:
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - alpha (float): the adjustment parameter, by default is 0.5
    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.

    In this example, the cpc_ds is passed as the first argument and is used as the reference 
    dataset. The imerg_ds is passed as the second argument and is used as the dataset to be 
    corrected. The adjustment parameter alpha is still optional and the default value is 0.5, 
    but the user can change it to another value. The corrected precipitation is then reshaped 
    back to the original shape before returning the corrected dataset.
    """
    from scipy.stats import percentileofscore

    # Get the precipitation data from the input datasets
    cpc_precip = cpc_ds['precip'].values
    imerg_precip = imerg_ds['precipitationCal'].values

    # Compute the percentiles of the observed and simulated data
    cpc_percentiles = np.array([percentileofscore(cpc_precip, val, kind='rank') for val in cpc_precip])
    imerg_percentiles = np.array([percentileofscore(imerg_precip, val, kind='rank') for val in imerg_precip])
    
    # Define the bias correction function
    def aQM_precip(cpc_precip, imerg_precip, alpha):
        cpc_percentiles = np.array([percentileofscore(cpc_precip, val, kind='rank') for val in cpc_precip])
        imerg_percentiles = np.array([percentileofscore(imerg_precip, val, kind='rank') for val in imerg_precip])
        corrected = np.interp(imerg_percentiles, (1-alpha)*cpc_percentiles + alpha*50, cpc_precip)
        
        return corrected

    corrected_precip = xr.apply_ufunc(aQM_precip, cpc_precip, imerg_precip, alpha, dask='parallelized')
    
    # Create a new xarray.Dataset with the bias-corrected data
    corrected_ds = xr.Dataset(data_vars={'precipitation': (('time', 'lat', 'lon'), corrected_precip)},
                              coords={'time': imerg_ds['time'],
                                      'lat': imerg_ds['lat'],
                                      'lon': imerg_ds['lon']})
    
    return corrected_ds

def gQM(cpc_ds, imerg_ds):
    """
    Correct the bias in the simulated data using Gamma Distribution Quantile Mapping (gQM) method.
    Parameters:
    - cpc_ds (xarray.Dataset): the observed data to use as a reference for correction.
    - imerg_ds (xarray.Dataset): the simulated data to be corrected.
    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected simulated data.

    In this example, the obs_ds is passed as the first argument and is used as the reference dataset. 
    The sim_ds is passed as the second argument and is used as the dataset to be corrected. T
    he scipy.stats.gamma.fit() function is used to fit the gamma distribution to the observed data, 
    and the scipy.stats.gamma.ppf() function is used to perform the bias correction by transforming 
    the percentiles of the simulated data to the percentiles of the observed data. The corrected 
    precipitation is then reshaped back to the original shape before returning the corrected dataset.
    """
    import scipy.stats as stats

    # Get the precipitation data from the input datasets
    cpc_precip = cpc_ds['precip'].values
    imerg_precip = imerg_ds['precipitationCal'].values

    # Fit a gamma distribution to the observed data
    cpc_shape, cpc_loc, cpc_scale = stats.gamma.fit(cpc_precip)
    
    # Define the bias correction function
    def gQM_precip(imerg, cpc_shape, cpc_loc, cpc_scale):
        corrected = stats.gamma.ppf(stats.rankdata(imerg_precip)/len(imerg_precip), cpc_shape, loc=cpc_loc, scale=cpc_scale)
        
        return corrected
        
    corrected_precip = xr.apply_ufunc(gQM_precip, imerg_precip, cpc_shape, cpc_loc, cpc_scale, dask='parallelized')
    
    # Create a new xarray.Dataset with the bias-corrected data
    corrected_ds = xr.Dataset(data_vars={'precipitation': (('time', 'lat', 'lon'), corrected_precip)},
                              coords={'time': imerg_ds['time'],
                                      'lat': imerg_ds['lat'],
                                      'lon': imerg_ds['lon']})
    
    return corrected_ds

def gpdQM(cpc_ds, imerg_ds):
    """
    Correct the bias in the simulated data using a combination of Gamma and Generalized Pareto Distribution Quantile Mapping (gpdQM) method.
    Parameters:
    - cpc_ds (xarray.Dataset): the observed data to use as a reference for correction.
    - imerg_ds (xarray.Dataset): the simulated data to be corrected.
    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected simulated data.

    the gpdQM_bias_correction() function first fits a gamma distribution to the observed data using scipy.stats.gamma.fit(), ]
    and a Generalized Pareto Distribution (GPD) to the tail of the observed data using scipy.stats.genpareto.fit(). 
    Then, it applies the bias correction using the percentile-percentile method, where percentiles of simulated 
    precipitation are transformed to percentiles of observed precipitation using the scipy.stats.genpareto.ppf() 
    and scipy.stats.gamma.ppf() functions.
    """
    import scipy.stats as stats

    # Get the precipitation data from the input datasets
    cpc_precip = cpc_ds['precip'].values
    imerg_precip = imerg_ds['precipitationCal'].values

    # Fit a gamma distribution to the observed data
    cpc_shape, cpc_loc, cpc_scale = stats.gamma.fit(cpc_precip)
    # Fit a Generalized Pareto Distribution to the tail of the observed data
    cpc_gpd_shape, cpc_gpd_loc, cpc_gpd_scale = stats.genpareto.fit(cpc_precip, floc=cpc_loc)
    
    # Define the bias correction function
    def gpdQM_precip(imerg_precip, cpc_shape, cpc_loc, cpc_scale, cpc_gpd_shape, cpc_gpd_loc, cpc_gpd_scale):
        
        # Use the gamma distribution for the majority of the data and the GPD for the tail
        threshold = stats.gamma.ppf(1-1e-3, cpc_shape, loc=cpc_loc, scale=cpc_scale)
        imerg_tail = imerg_precip[imerg_precip > threshold]
        imerg_body = imerg_precip[imerg_precip <= threshold]
        corrected_tail = stats.genpareto.ppf(stats.rankdata(imerg_tail)/len(imerg_tail), cpc_gpd_shape, loc=cpc_gpd_loc, scale=cpc_gpd_scale)
        corrected_body = stats.gamma.ppf(stats.rankdata(imerg_body)/len(imerg_body), cpc_shape, loc=cpc_loc, scale=cpc_scale)
        corrected = np.concatenate((corrected_body, corrected_tail))
        
        return corrected
        
    corrected_precip = xr.apply_ufunc(gpdQM_precip, imerg_precip, cpc_shape, cpc_loc, cpc_scale, cpc_gpd_shape, cpc_gpd_loc, cpc_gpd_scale, dask='parallelized')
    
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

        # Get the number of days in dekad
        start_date = corrected_ds['time'][start_idx].values
        end_date = corrected_ds['time'][end_idx].values + np.timedelta64(1,'D')
        num_days = (end_date-start_date).astype('timedelta64[D]') / np.timedelta64(1,'D')

        # Divide the mean by the number of days in this dekad, accounting for leap years
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
    imerg_corrected = imerg_ds['precipitation']
    cpc_precip = cpc_ds['precip']

    # Calculate the relative bias
    relative_bias = (imerg_corrected.sum(dim='time') / cpc_precip.sum(dim='time')).mean(dim=('lat', 'lon'))

    # Calculate the Pearson correlation coefficient
    pearson = imerg_corrected.corr(cpc_precip, dim='time')

    # Calculate the root mean squared error
    rmse = (((imerg_corrected - cpc_precip)**2).mean(dim='time')**0.5).mean(dim=('lat', 'lon'))

    # Calculate the mean absolute error
    mae = (np.abs(imerg_corrected - cpc_precip)).mean(dim='time').mean(dim=('lat', 'lon'))

    # Calculate the probability of detection
    pod = (imerg_corrected > 0).sum(dim='time') / (cpc_precip > 0).sum(dim='time')

    # Calculate the false alarm ratio
    far = ((imerg_corrected > 0) & (cpc_precip == 0)).sum(dim='time') / (cpc_precip == 0).sum(dim='time')

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

def main(run_all_methods=False, method=None):
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
    if not run_all_methods:
        
        if method is None:
            method = bias_correction.__defaults__[0]
    
    else:
        method = None

    # Define the appropriate input and output directory paths
    input_dir = f'input'
    imerg_path = f'{input_dir}/imerg'
    cpc_path = f'{input_dir}/cpc'
    output_dir = f'output'
    # method_dir = f'output/{method}'
    # corrected_path = f'{method_dir}/corrected'
    # factors_path = f'{method_dir}/factors'
    # metrics_path = f'{method_dir}/metrics'

    # Create the output directories if they don't already exist
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(method_dir, exist_ok=True)
    # os.makedirs(corrected_path, exist_ok=True)
    # os.makedirs(factors_path, exist_ok=True)
    # os.makedirs(metrics_path, exist_ok=True)

    # Load the IMERG data
    imerg_ds = xr.open_mfdataset('{imerg_path}/*.nc')

    # Load the CPC data
    cpc_ds = xr.open_mfdataset('{cpc_path}/*.nc')

    # Initialize an empty dataframe to store the metric values
    metrics = pd.DataFrame(columns=['relative_bias', 'pearson', 'rmse', 'mae', 'pod', 'far', 'csi'])

    # Use dask's delayed function to schedule the computation for each year in parallel
    corrected_ds_list = []
    
    if run_all_methods:
        methods = ['scale', 'distribution', 'delta', 'lsc', 'lscdf', 'rcdfm', 'mlr', 
        'ann', 'kalman', 'bcsd', 'bcsdm', 'qq_mapping', 'eQM', 'aQM', 'gQM', 'gpdQM']
    
    else:
        methods = [method]
    
    for method in methods:
        os.makedirs(f'output/{method}', exist_ok=True)
        os.makedirs(f'output/{method}/corrected', exist_ok=True)
        os.makedirs(f'output/{method}/factors', exist_ok=True)
        os.makedirs(f'output/{method}/metrics', exist_ok=True)
        
        for year in range(2001, 2023):
            # Get the data for this year
            imerg_year_ds = imerg_ds.sel(time=imerg_ds['time.year'] == year)
            cpc_year_ds = cpc_ds.sel(time=cpc_ds['time.year'] == year)

            # Schedule the calculation of the corrected data for this year
            corrected_ds = delayed(bias_correction)(imerg_year_ds, cpc_year_ds, method=method)
            corrected_ds_list.append(corrected_ds)

    # Compute the corrected data for all years in parallel
    corrected_ds_list = dask.compute(*corrected_ds_list)

    """
    The `zip` function is used here to iterate over the three lists `methods`, `range(2001, 2023)`, and 
    `corrected_ds_list` simultaneously. The `*` operator is used to repeat the elements of the `methods` 
    list, so that there is one method for each year. The `*len(methods)` is used to repeat the elements 
    of the `range(2001, 2023)` list, so that there is one year for each method. This way, for each 
    iteration of the loop, the variable `method` will be set to the current method, the variable `year` 
    will be set to the current year, and the variable `corrected_ds` will be set to the corrected data 
    for that year and method.
    """
    # Save the corrected data to a NetCDF file
    for method, year, corrected_ds in zip(methods*(2022-2001+1), range(2001, 2023)*len(methods), corrected_ds_list):
        corrected_ds.to_netcdf(f'output/{method}/corrected/corrected_{method}_{year}.nc')
        
        dekad_factors = create_multiplying_factors(corrected_ds)
        
        for i, factor in enumerate(dekad_factors):
            factor.to_netcdf(f'output/{method}/factors/multiplying_factor_{method}_{i+1}.nc')

        # Calculate the metrics for the corrected data
        year_metrics = calculate_metrics(corrected_ds, cpc_year_ds)

        # Add the metric values for this year to the dataframe
        metrics = metrics.append(year_metrics)

    # Save the metrics dataframe to a CSV file
    if run_all_methods:
        metrics.to_csv(f'output/{method}_metrics.csv')
    
    else:
        metrics.to_csv(f'output/{method}/metrics/{method}_metrics.csv')

if __name__ == '__main__':
    main(run_all_methods=True) # run all methods
    # main(method='rcdfm') # run rcdfm method
