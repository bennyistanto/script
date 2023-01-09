# -*- coding: utf-8 -*-
"""
NAME
    imerg_cpc_edqm.py
    Bias correction using the Equal Distance-based CDF Quantile Mapping Methods
DESCRIPTION
    Input data for this script will use GPM IMERG daily data compiled as 1 year 1 nc file
    and CPC Global Unified Gauge-Based Analysis of Daily Precipitation (GUGBADP).
    This script can do a bias correction using the Equal Distance-based CDF Quantile Mapping 
    Methods, which consist of: (1) eQM Empirical Quantile Mapping, (2) aQM adjusted Quantile 
    Mapping, (3) gQM Parametric Quantile Mapping: Gamma Distribution and (4) gpQM Parametric 
    Quantile Mapping: Gamma and Generalized Pareto Distribution
REQUIREMENT
    It required numpy, scipy, sklearn and xarray module. So it will work on any machine environment
EXAMPLES
    python imerg_cpc_edqm.py
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
from scipy.stats import gamma, gennorm
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def eQM(obs, sim):
    """Empirical Quantile Mapping"""
    # Sort the data
    obs_sorted = np.sort(obs)
    sim_sorted = np.sort(sim)
    
    # Find the index of the nearest value in the sorted simulated data for each value in the sorted observed data
    sim_index = np.searchsorted(sim_sorted, obs_sorted)
    
    # Replace the values in the simulated data with the corresponding values in the observed data
    sim_sorted[sim_index] = obs_sorted
    
    # Interpolate to get the values for the original indices
    corrected = np.interp(sim, sim_sorted, obs_sorted)
    
    return corrected

def aQM(obs, sim, alpha=1):
    """Adjusted Quantile Mapping"""
    # Sort the data
    obs_sorted = np.sort(obs)
    sim_sorted = np.sort(sim)
    
    # Find the index of the nearest value in the sorted simulated data for each value in the sorted observed data
    sim_index = np.searchsorted(sim_sorted, obs_sorted)
    
    # Replace the values in the simulated data with the corresponding values in the observed data
    sim_sorted[sim_index] = obs_sorted
    
    # Get the quantiles of the sorted observed data and the sorted simulated data
    obs_quantiles = np.arange(len(obs)) / len(obs)
    sim_quantiles = np.arange(len(sim)) / len(sim)
    
    # Find the quantiles of the sorted simulated data at the indices of the sorted observed data
    sim_quantiles_at_obs_index = sim_quantiles[sim_index]
    
    # Adjust the quantiles of the sorted simulated data using the formula:
    # adjusted_quantile = (quantile - alpha * (quantile - empirical_quantile)) / (1 - alpha * (quantile - empirical_quantile))
    adjusted_quantiles = (sim_quantiles_at_obs_index - alpha * (sim_quantiles_at_obs_index - obs_quantiles)) / (1 - alpha * (sim_quantiles_at_obs_index - obs_quantiles))
    
    # Interpolate to get the values for the original indices
    corrected = np.interp(sim, sim_sorted, obs_sorted, left=np.nan, right=np.nan)
    
    return corrected

def gQM(obs, sim, params=None):
    """Parametric Quantile Mapping: Gamma Distribution"""
    # Sort the data
    obs_sorted = np.sort(obs)
    sim_sorted = np.sort(sim)
    
    # Find the index of the nearest value in the sorted simulated data for each value in the sorted observed data
    sim_index = np.searchsorted(sim_sorted, obs_sorted)
    
    # Replace the values in the simulated data with the corresponding values in the observed data
    sim_sorted[sim_index] = obs_sorted
    
    # Get the quantiles of the sorted observed data and the sorted simulated data
    obs_quantiles = np.arange(len(obs)) / len(obs)
    sim_quantiles = np.arange(len(sim)) / len(sim)
    
    # Find the quantiles of the sorted simulated data at the indices of the sorted observed data
    sim_quantiles_at_obs_index = sim_quantiles[sim_index]
    
    # Fit a gamma distribution to the quantiles
    if params is None:
        params, _ = gamma.fit(sim_quantiles_at_obs_index, floc=0)
    
    # Use the fitted distribution to transform the quantiles of the simulated data
    transformed_quantiles = gamma.cdf(sim_quantiles, *params)
    
    # Interpolate to get the values for the original indices
    corrected = np.interp(sim, sim_sorted, obs_sorted, left=np.nan, right=np.nan)
    
    return corrected

def gpQM(obs, sim, params=None):
    """Parametric Quantile Mapping: Gamma and Generalized Pareto Distribution"""
    # Sort the data
    obs_sorted = np.sort(obs)
    sim_sorted = np.sort(sim)
    
    # Find the index of the nearest value in the sorted simulated data for each value in the sorted observed data
    sim_index = np.searchsorted(sim_sorted, obs_sorted)
    
    # Replace the values in the simulated data with the corresponding values in the observed data
    sim_sorted[sim_index] = obs_sorted
    
    # Get the quantiles of the sorted observed data and the sorted simulated data
    obs_quantiles = np.arange(len(obs)) / len(obs)
    sim_quantiles = np.arange(len(sim)) / len(sim)
    
    # Find the quantiles of the sorted simulated data at the indices of the sorted observed data
    sim_quantiles_at_obs_index = sim_quantiles[sim_index]
    
    # Fit a generalized Pareto distribution to the right tail of the quantiles
    if params is None:
        k, sigma, loc = gennorm.fit(sim_quantiles_at_obs_index[sim_quantiles_at_obs_index > np.percentile(sim_quantiles_at_obs_index, 90)], floc=0)
    
        # Fit a gamma distribution to the left tail of the quantiles
        a, loc, scale = gamma.fit(sim_quantiles_at_obs_index[sim_quantiles_at_obs_index <= np.percentile(sim_quantiles_at_obs_index, 90)], floc=0)
    
    # Use the fitted distributions to transform the quantiles of the simulated data
    transformed_quantiles = np.empty_like(sim_quantiles)
    transformed_quantiles[sim_quantiles <= np.percentile(sim_quantiles, 90)] = gamma.cdf(sim_quantiles[sim_quantiles <= np.percentile(sim_quantiles, 90)], a, loc, scale)
    transformed_quantiles[sim_quantiles > np.percentile(sim_quantiles, 90)] = gennorm.cdf(sim_quantiles[sim_quantiles > np.percentile(sim_quantiles, 90)], k, sigma, loc)
    
    # Interpolate to get the values for the original indices
    corrected = np.interp(sim, sim_sorted, obs_sorted, left=np.nan, right=np.nan)
    
    return corrected

def calculate_metrics(obs, sim):
    """Calculate the relative bias, Pearson correlation coefficient, root mean squared error, mean absolute error, probability of detection, false alarm rate, and critical success index"""
    relative_bias = np.mean(sim - obs) / np.mean(obs)
    pearson = np.corrcoef(obs, sim)[0, 1]
    rmse = np.sqrt(mean_squared_error(obs, sim))
    mae = mean_absolute_error(obs, sim)
    pod = np.sum((sim > 0) & (obs > 0)) / np.sum(obs > 0)
    far = np.sum((sim > 0) & (obs == 0)) / np.sum(obs == 0)
    csi = np.sum((sim > 0) & (obs > 0)) / (np.sum((sim > 0) & (obs > 0)) + np.sum((sim > 0) & (obs == 0)) + np.sum((sim == 0) & (obs > 0)))
    
    return relative_bias, pearson, rmse, mae, pod, far, csi

def bias_correction(obs_ds, sim_ds, method, params=None):
    """Perform bias correction using one of the four available methods"""
    # Get the data arrays
    obs = obs_ds.values
    sim = sim_ds.values
    
    # Choose the method
    if method == "eQM":
        corrected = eQM(obs, sim)
    elif method == "aQM":
        corrected = aQM(obs, sim, params)
    elif method == "gQM":
        corrected = gQM(obs, sim, params)
    elif method == "gpQM":
        corrected = gpQM(obs, sim, params)
    
    # Calculate the metrics
    relative_bias, pearson, rmse, mae, pod, far, csi = calculate_metrics(obs, corrected)
    
    # Return the corrected data and the metrics
    return corrected, relative_bias, pearson, rmse, mae, pod, far, csi

def calculate_multiplying_factor(obs_ds, sim_ds, method, params=None):
    # Convert the data to pandas DataFrames
    obs_df = obs_ds.to_dataframe()
    sim_df = sim_ds.to_dataframe()
    
    # Group the data by dekad
    obs_df = obs_df.resample("10D").sum()
    sim_df = sim_df.resample("10D").sum()
    
    # Calculate the multiplying factor
    if method == "eQM":
        # Calculate the quantiles for the observed data
        obs_quantiles = obs_df.quantile(np.arange(0, 1.1, 0.1), interpolation="linear")
        
        # Calculate the quantiles for the simulated data
        sim_quantiles = sim_df.quantile(np.arange(0, 1.1, 0.1), interpolation="linear")
        
        # Calculate the multiplying factor
        multiplying_factor = sim_quantiles.values / obs_quantiles.values
    elif method == "aQM":
        # Get the lower and upper bounds for the quantiles
        lower_bound, upper_bound = params
        
        # Calculate the quantiles for the observed data
        obs_quantiles = obs_df.quantile(np.arange(lower_bound, upper_bound + 0.1, 0.1), interpolation="linear")
        
        # Calculate the quantiles for the simulated data
        sim_quantiles = sim_df.quantile(np.arange(lower_bound, upper_bound + 0.1, 0.1), interpolation="linear")
        
        # Calculate the multiplying factor
        multiplying_factor = sim_quantiles.values / obs_quantiles.values
    elif method == "gQM":
        # Fit a gamma distribution to the left tail of the quantiles
        shape, loc, scale = stats.gamma.fit(obs_df.values.flatten())
        
        # Calculate the quantiles for the observed data
        obs_quantiles = obs_df.quantile(stats.gamma.cdf(np.arange(0, obs_df.max().max() + 1, 1), shape, loc, scale), interpolation="linear")
        
        # Calculate the quantiles for the simulated data
        sim_quantiles = sim_df.quantile(stats.gamma.cdf(np.arange(0, obs_df.max().max() + 1, 1), shape, loc, scale), interpolation="linear")
        
        # Calculate the multiplying factor
        multiplying_factor = sim_quantiles.values / obs_quantiles.values
    elif method == "gpQM":
        # Get the lower and upper bounds for the quantiles
        lower_bound, upper_bound = params
        
        # Fit a generalized Pareto distribution to the right tail of the quantiles
        shape, loc, scale = stats.genpareto.fit(obs_df.values.flatten(), floc=0)

def main():
    # Set the path to the observed and simulated data
    obs_path = "./obs/"
    sim_path = "./sim/"
    
    # Set the names of the variables in the NetCDF files
    obs_var_name = "precipitation"
    sim_var_name = "precipitation"
    
    # Set the start and end years for the loop
    start_year = 2001
    end_year = 2022
    
    # Set the bias correction method
    method = "gQM"
    
    # Set the parameters for the aQM and gpQM methods
    params = None
    if method == "aQM":
        params = (0.1, 0.9)
    elif method == "gpQM":
        params = (0.1, 0.9)
    
    # Create a list to store the results
    results = []
    
    # Loop through the years
    for year in range(start_year, end_year + 1):
        # Load the observed and simulated data for the year
        obs_ds = xr.open_dataset(obs_path + "imerg_{}.nc".format(year), decode_cf=True)[obs_var_name]
        sim_ds = xr.open_dataset(sim_path + "cpc_{}.nc".format(year), decode_cf=True)[sim_var_name]
        
        # Perform bias correction
        corrected, relative_bias, pearson, rmse, mae, pod, far, csi = bias_correction(obs_ds, sim_ds, method, params)
        
        # Save the corrected data to a NetCDF file
        corrected_ds = xr.Dataset({"precipitation": (["time", "lat", "lon"], corrected)}, coords={"time": obs_ds["time"], "lat": obs_ds["lat"], "lon": obs_ds["lon"]})
        corrected_ds.to_netcdf("./corrected/{}_{}.nc".format(method, year))
        
        # Append the results to the list
        results.append([year, relative_bias, pearson, rmse, mae, pod, far, csi])
    
    # Convert the list to a DataFrame and save it to a CSV file
    results_df = pd.DataFrame(results, columns=["year", "relative_bias", "pearson", "rmse", "mae", "pod", "far", "csi"])
    results_df.to_csv("./results/{}.csv".format(method), index=False)
    
    # Calculate the multiplying factor for dekad-level bias correction
    multiplying_factor = calculate_multiplying_factor(obs_ds, sim_ds, method, params)

    # Calculate the number of days in each dekad
    days_per_dekad = obs_df.index.days_in_month / 3

    # Divide the multiplying factor by the number of days in each dekad
    multiplying_factor /= days_per_dekad
    
    # Save the multiplying factor to a NetCDF file
    multiplying_factor_ds = xr.Dataset({"multiplying_factor": (["time"], multiplying_factor
