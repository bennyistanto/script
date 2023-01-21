# -*- coding: utf-8 -*-
"""
NAME
    ensemble_corrected_data.py
DESCRIPTION
    Ensemble bias correction output using various methods:
    i. Decision Tree
    ii. Bayesian
    iii. Random Forest
    iv. Neural Network
    v. Boosting
REQUIREMENT
    It required os, calendar, numpy, xarray, pandas, scipy, and dask module. 
    So it will work on any machine environment. 
    And please do check specific module in every function.
HOW-TO USE
    python ensemble_corrected_data.py
NOTES
    Input data for this script will use corrected precipitation output data from
    imerg_cpc_biascorrection.py process and the CPC Global Unified Gauge-Based 
    Analysis of Daily Precipitation (GUGBADP).
WORKING DIRECTORY
    /input/cpc - put your GUGBADP data here
    /output/{method}/corrected - location for corrected precipitation, and serve as input
    /output/ensemble/corrected - location for ensemble corrected precipitation output
    /output/ensemble/factors - location for corrected multiplying factors output
    /output/ensemble/metrics - location for corrected statistical metrics output
DATA
    IMERG Corrected Data: /output/{method}/corrected
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

def ensemble_method(corrected_ds_list, cpc_ds, method='decisiontree'):
    """
    Correct the bias in the IMERG data using the Replacement-based CDF Mapping method or other methods.

    Parameters:
    - imerg_ds (xarray.Dataset): the IMERG data to be corrected.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for correction.
    - method (str): the method to use for correction, either 'rcdfm', 'scale', 'distribution' or other.

    Returns:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.
    """
    if method == 'decisiontree':
        corrected_ds = decisiontree(corrected_ds_list, cpc_ds, method='mean')
    elif method == 'bayesian':
        corrected_ds = bayesian(corrected_ds_list, cpc_ds)
    elif method == 'randomforest':
        corrected_ds = randomforest(corrected_ds_list, cpc_ds)
    elif method == 'neuralnetwork':
        corrected_ds = neuralnetwork(corrected_ds_list, cpc_ds)
    elif method == 'boost':
        corrected_ds = boost(corrected_ds_list, cpc_ds)
    # add more elif statement for other correction methods
        else:
        raise ValueError("Invalid method. Choose either 'decisiontree', 'bayesian', 'randomforest', 'neuralnetwork', 'boost'.")

    return corrected_ds


def decisiontree(corrected_ds_list, cpc_ds, method_names):
    """
    Ensemble the corrected precipitation data using decision trees.
    Parameters:
    - corrected_ds_list (list of xarray.Dataset): the bias-corrected precipitation data.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for evaluation.
    - method_names (list of str): the names of the methods used to correct the data.
    Returns:
    - ensemble_ds (xarray.Dataset): the ensemble of the bias-corrected precipitation data.

    An ensemble method that uses decision trees to learn the optimal weights for combining the bias-corrected 
    precipitation data from multiple methods. The weights are calculated by training a decision tree model on 
    the error between the corrected data and the reference CPC data. The weights are then normalized using 
    the normalize_weights() function, which I included in the code, to ensure that they sum to 1. The corrected 
    data is then multiplied by the weights and summed to create the ensemble precipitation data. Finally, 
    the function returns the ensemble precipitation data in the form of an xarray dataset.
    """
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    # Combine the corrected data from each method into a single xarray.Dataset
    ensemble_data = xr.concat(corrected_ds_list, dim='method')
    ensemble_ds = xr.Dataset(data_vars={'precipitation': (('method', 'time', 'lat', 'lon'), ensemble_data)},
                             coords={'method': method_names,
                                     'time': ensemble_data['time'],
                                     'lat': ensemble_data['lat'],
                                     'lon': ensemble_data['lon']})
    
    # Get the cpc data as a 1-D array
    cpc_data = cpc_ds['precip'].values.ravel()
    
    # Initialize a list to store the error of each method
    errors = []
    for method in method_names:
        
        # Get the corrected data for this method as a 1-D array
        corrected_data = ensemble_ds['precipitation'].sel(method=method).values.ravel()
        
        # Calculate the error of this method
        error = np.mean(np.abs(cpc_data - corrected_data))
        errors.append(error)
    
    
    def normalize_weights(weights):
        """
        Normalize the weights so that they sum up to 1.
        Parameters:
        - weights (numpy.array): the weights to normalize.
        Returns:
        - normalized_weights (numpy.array): the normalized weights.
        """
        normalized_weights = weights / np.sum(weights)
        return normalized_weights
    
    # Normalize the errors
    errors = normalize_weights(errors)
    
    # Create decision tree regressor to learn the optimal weights
    regressor = DecisionTreeRegressor()
    
    # Fit the regressor with the errors and method names as input and target
    regressor.fit(np.array(method_names).reshape(-1,1), errors)
    
    # Get the predicted weights
    weights = regressor.predict(np.array(method_names).reshape(-1,1))
    
    # Assign weights to the ensemble_ds
    ensemble_ds['precipitation'] = ensemble_ds['precipitation'].assign_coords(weight=weights)

    # Multiply the corrected data with the weights
    ensemble_ds['precipitation'] = ensemble_ds['precipitation'] * ensemble_ds['precipitation'].weight

    # Sum the weighted corrected data to get the ensemble precipitation
    ensemble_ds = ensemble_ds.sum(dim='method')
    
    return ensemble_ds

def bayesian(corrected_ds_list, cpc_ds, method_names):
    """
    Ensemble the corrected precipitation data using Bayesian Ridge Regression.
    Parameters:
    - corrected_ds_list (list of xarray.Dataset): the bias-corrected precipitation data.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for evaluation.
    - method_names (list of str): the names of the methods used to correct the data.
    Returns:
    - ensemble_ds (xarray.Dataset): the ensemble of the bias-corrected precipitation data.
    
    an ensemble method that uses a Bayesian Ridge Regression model to learn the optimal weights for 
    combining the bias-corrected precipitation data from multiple methods. The weights are calculated by 
    training the model on the error between the corrected data and the reference CPC data. The corrected 
    data is then multiplied by the weights and summed to create the ensemble precipitation data. 
    The function returns the ensemble precipitation data in the form of an xarray dataset.
    """
    from sklearn.linear_model import BayesianRidge
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    # Combine the corrected data from each method into a single xarray.Dataset
    ensemble_data = xr.concat(corrected_ds_list, dim='method')
    ensemble_ds = xr.Dataset(data_vars={'precipitation': (('method', 'time', 'lat', 'lon'), ensemble_data)},
                             coords={'method': method_names,
                                     'time': ensemble_data['time'],
                                     'lat': ensemble_data['lat'],
                                     'lon': ensemble_data['lon']})
    
    # Define the feature and target arrays
    X = np.array([corrected_ds_list[i]['precipitation'].values for i in range(len(corrected_ds_list))])
    Y = cpc_ds['precip'].values

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # Train the Bayesian Ridge Regression model
    br = BayesianRidge()
    br.fit(X_train, Y_train)
    
    # Use the trained model to predict the error for each method
    errors = mean_squared_error(Y_test, br.predict(X_test), multioutput='raw_values')
    
    def normalize_weights(weights):
        """
        Normalize the weights so that they add up to 1.
        Parameters:
        - weights (list): the weights to normalize.
        Returns:
        - normalized_weights (list): the normalized weights.
        """
        total = sum(weights)
        normalized_weights = [weight/total for weight in weights]
        
        return normalized_weights

    # Normalize the errors
    weights = normalize_weights(errors)
    
    # Assign weights to the ensemble_ds
    ensemble_ds['precipitation'] = ensemble_ds['precipitation'].assign_coords(weight=weights)
    
    # Multiply the corrected data with the weights
    ensemble_ds['precipitation'] = ensemble_ds['precipitation'] * ensemble_ds['precipitation'].weight
    
    # Sum the weighted corrected data to get the ensemble precipitation
    ensemble_ds = ensemble_ds.sum(dim='method')
    
    return ensemble_ds

def randomforest(corrected_ds_list, cpc_ds, method_names):
    """
    Ensemble the corrected precipitation data using Random Forest.
    Parameters:
    - corrected_ds_list (list of xarray.Dataset): the bias-corrected precipitation data.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for evaluation.
    - method_names (list of str): the names of the methods used to correct the data.
    Returns:
    - ensemble_ds (xarray.Dataset): the ensemble of the bias-corrected precipitation data.

    An ensemble method that uses random forest to learn the optimal weights for combining the bias-corrected 
    precipitation data from multiple methods. The weights are calculated by training a random forest model on 
    the error between the corrected data and the reference CPC data. The weights are then normalized using 
    the normalize_weights() function, which I included in the code, to ensure that they sum to 1. The corrected 
    data is then multiplied by the weights and summed to create the ensemble precipitation data. Finally, 
    the function returns the ensemble precipitation data in the form of an xarray dataset.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    # Combine the corrected data from each method into a single xarray.Dataset
    ensemble_data = xr.concat(corrected_ds_list, dim='method')
    ensemble_ds = xr.Dataset(data_vars={'precipitation': (('method', 'time', 'lat', 'lon'), ensemble_data)},
                             coords={'method': method_names,
                                     'time': ensemble_data['time'],
                                     'lat': ensemble_data['lat'],
                                     'lon': ensemble_data['lon']})
    
    # Define the feature and target arrays
    X = np.array([corrected_ds_list[i]['precipitation'].values for i in range(len(corrected_ds_list))])
    Y = cpc_ds['precip'].values

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    # Train the Random Forest model
    rf = RandomForestRegressor()
    rf.fit(X_train, Y_train)

    # Calculate the error of each method
    errors = []
    
    for method in method_names:
        # Get the corrected data for this method as a 1-D array
        corrected_data = ensemble_ds['precipitation'].sel(method=method).values.ravel()
        
        # Calculate the error of this method
        error = mean_squared_error(cpc_ds['precip'].values.ravel(), corrected_data)
        errors.append(error)
    
    def normalize_weights(weights):
        """
        Normalize the weights so they sum to 1.
        Parameters:
        - weights (list or numpy array): the weights to normalize.
        Returns:
        - normalized_weights (numpy array): the normalized weights.
        """
        normalized_weights = weights / np.sum(weights)
        
        return normalized_weights

    # Normalize the errors
    errors = normalize_weights(errors)
    
    # Use the trained model to predict the optimal weights for new data
    weights = rf.predict(X_test)
    weights = normalize_weights(weights)

    # Assign weights to the ensemble_ds
    ensemble_ds['precipitation'] = ensemble_ds['precipitation'].assign_coords(weight=weights)

    # Multiply the corrected data with the weights
    ensemble_ds['precipitation'] = ensemble_ds['precipitation'] * ensemble_ds['precipitation'].weight

    # Sum the weighted corrected data to get the ensemble precipitation
    ensemble_ds = ensemble_ds.sum(dim='method')
    
    return ensemble_ds

def neuralnetwork(corrected_ds_list, cpc_ds, method_names):
    """
    Ensemble the corrected precipitation data using Neural Networks.
    Parameters:
    - corrected_ds_list (list of xarray.Dataset): the bias-corrected precipitation data.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for evaluation.
    - method_names (list of str): the names of the methods used to correct the data.
    Returns:
    - ensemble_ds (xarray.Dataset): the ensemble of the bias-corrected precipitation data.

    An ensemble method that uses a neural network to learn the optimal weights for combining the bias-corrected 
    precipitation data from multiple methods. The network is trained on the error between the corrected data and 
    the reference CPC data, and the weights are calculated based on the network's predictions. The corrected data 
    is then multiplied by the weights and summed to create the ensemble precipitation data.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    import xarray as xr
    import numpy as np

    # Combine the corrected data from each method into a single xarray.Dataset
    ensemble_data = xr.concat(corrected_ds_list, dim='method')
    ensemble_ds = xr.Dataset(data_vars={'precipitation': (('method', 'time', 'lat', 'lon'), ensemble_data)},
                             coords={'method': method_names,
                                     'time': ensemble_data['time'],
                                     'lat': ensemble_data['lat'],
                                     'lon': ensemble_data['lon']})
    
    # Define the feature and target arrays
    X = np.array([corrected_ds_list[i]['precipitation'].values for i in range(len(corrected_ds_list))])
    Y = cpc_ds['precip'].values

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    # Define the model
    nn = MLPRegressor(hidden_layer_sizes=(50,50,50), max_iter=500, activation = 'relu', solver='adam', random_state=1)
    nn.fit(X_train, Y_train)

    # Calculate the error of each method
    errors = []
    for method in method_names:
        # Get the corrected data for this method as a 1-D array
        corrected_data = ensemble_ds['precipitation'].sel(method=method).values.ravel()
        # Calculate the error of this method
        error = mean_squared_error(cpc_ds['precip'].values.ravel(), corrected_data)
        errors.append(error)
    
    def normalize_weights(weights):
        """
        Normalize the weights so they sum to 1.
        Parameters:
        - weights (list or numpy array): the weights to normalize.
        Returns:
        - normalized_weights (numpy array): the normalized weights.
        """
        normalized_weights = weights / np.sum(weights)
        
        return normalized_weights

    # Normalize the errors
    errors = normalize_weights(errors)
    
    # Use the trained model to predict the optimal weights for new data
    weights = nn.predict(X_test)
    weights = normalize_weights(weights)

    # Assign weights to the ensemble_ds
    ensemble_ds['precipitation'] = ensemble_ds['precipitation'].assign_coords(weight=weights)

    # Multiply the corrected data with the weights
    ensemble_ds['precipitation'] = ensemble_ds['precipitation'] * ensemble_ds['precipitation'].weight

    # Sum the weighted corrected data to get the ensemble precipitation
    ensemble_ds = ensemble_ds.sum(dim='method')
    
    return ensemble_ds

def boost(corrected_ds_list, cpc_ds, method_names):
    """
    Ensemble the corrected precipitation data using Boosting.
    Parameters:
    - corrected_ds_list (list of xarray.Dataset): the bias-corrected precipitation data.
    - cpc_ds (xarray.Dataset): the CPC data to use as a reference for evaluation.
    - method_names (list of str): the names of the methods used to correct the data.
    Returns:
    - ensemble_ds (xarray.Dataset): the ensemble of the bias-corrected precipitation data.

    An ensemble method that uses a series of weak models, such as decision trees, to learn the optimal weights 
    for combining the bias-corrected precipitation data from multiple methods. The weights are calculated by 
    training a boosting model on the error between the corrected data and the reference CPC data. The corrected 
    data is then multiplied by the weights and summed to create the ensemble precipitation data.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    import xarray as xr
    import numpy as np
    
    # Combine the corrected data from each method into a single xarray.Dataset
    ensemble_data = xr.concat(corrected_ds_list, dim='method')
    ensemble_ds = xr.Dataset(data_vars={'precipitation': (('method', 'time', 'lat', 'lon'), ensemble_data)},
                             coords={'method': method_names,
                                     'time': ensemble_data['time'],
                                     'lat': ensemble_data['lat'],
                                     'lon': ensemble_data['lon']})
    
    # Define the feature and target arrays
    X = np.array([corrected_ds_list[i]['precipitation'].values for i in range(len(corrected_ds_list))])
    Y = cpc_ds['precip'].values

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    # Train the Boosting model
    boosting = GradientBoostingRegressor()
    boosting.fit(X_train, Y_train)

    # Calculate the error of each method
    errors = []
    for method in method_names:
        # Get the corrected data for this method as a 1-D array
        corrected_data = ensemble_ds['precipitation'].sel(method=method).values.ravel()
        
        # Calculate the error of this method
        error = mean_squared_error(cpc_ds['precip'].values.ravel(), corrected_data)
        errors.append(error)
    
    def normalize_weights(weights):
        """
        Normalize the weights so they sum to 1.
        Parameters:
        - weights (list or numpy array): the weights to normalize.
        Returns:
        - normalized_weights (numpy array): the normalized weights.
        """
        normalized_weights = weights / np.sum(weights)
        
        return normalized_weights

    # Normalize the errors
    errors = normalize_weights(errors)

    # Use the trained model to predict the optimal weights for new data
    weights = boosting.predict(X_test)
    weights = normalize_weights(weights)

    # Assign weights to the ensemble_ds
    ensemble_ds['precipitation'] = ensemble_ds['precipitation'].assign_coords(weight=weights)

    # Multiply the corrected data with the weights
    ensemble_ds['precipitation'] = ensemble_ds['precipitation'] * ensemble_ds['precipitation'].weight

    # Sum the weighted corrected data to get the ensemble precipitation
    ensemble_ds = ensemble_ds.sum(dim='method')

    return ensemble_ds

def create_multiplying_factors(ensemble_ds, num_dekads=36):
    """
    Create multiplying factors for correcting the IMERG data in the future.

    Parameters:
    - corrected_ds (xarray.Dataset): the bias-corrected IMERG data.
    - num_dekads (int): the number of dekads (10-day periods) to create multiplying factors for.

    Returns:
    - dekad_factors (list of xarray.DataArray): the multiplying factors for each dekad.
    """
    # Compute the number of days in each dekad
    days_per_dekad = ensemble_ds.time.size // num_dekads
    
    # Initialize the list to store the multiplying factors
    dekad_factors = []

    # Loop through each dekad
    for i in range(num_dekads):
        
        # Get the start and end indices for this dekad
        start_idx = i * days_per_dekad
        end_idx = start_idx + days_per_dekad - 1

        # Get the data for this dekad
        dekad_data = ensemble_ds['precipitation'].isel(time=slice(start_idx, end_idx+1))

        # Compute the mean of the data for this dekad
        dekad_mean = dekad_data.mean(dim='time')

        # Get the number of days in dekad
        start_date = ensemble_ds['time'][start_idx].values
        end_date = ensemble_ds['time'][end_idx].values + np.timedelta64(1,'D')
        num_days = (end_date-start_date).astype('timedelta64[D]') / np.timedelta64(1,'D')

        # Divide the mean by the number of days in this dekad, accounting for leap years
        dekad_factor = dekad_mean / num_days

        # Add the multiplying factor for this dekad to the list
        dekad_factors.append(dekad_factor)
    
    return dekad_factors

def calculate_metrics(ensemble_ds, cpc_ds):
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
    - ensemble_ds (xarray.Dataset): the IMERG corrected ensemble data.
    - cpc_ds (xarray.Dataset): the CPC data.

    Returns:
    - metrics (pandas.DataFrame): a dataframe containing the metric values.
    """
    # Get the precipitation data from the input datasets
    ensemble_precip = ensemble_ds['precipitation']
    cpc_precip = cpc_ds['precip']

    # Calculate the relative bias
    relative_bias = (ensemble_precip.sum(dim='time') / cpc_precip.sum(dim='time')).mean(dim=('lat', 'lon'))

    # Calculate the Pearson correlation coefficient
    pearson = ensemble_precip.corr(cpc_precip, dim='time')

    # Calculate the root mean squared error
    rmse = (((ensemble_precip - cpc_precip)**2).mean(dim='time')**0.5).mean(dim=('lat', 'lon'))

    # Calculate the mean absolute error
    mae = (np.abs(ensemble_precip - cpc_precip)).mean(dim='time').mean(dim=('lat', 'lon'))

    # Calculate the probability of detection
    pod = (ensemble_precip > 0).sum(dim='time') / (cpc_precip > 0).sum(dim='time')

    # Calculate the false alarm ratio
    far = ((ensemble_precip > 0) & (cpc_precip == 0)).sum(dim='time') / (cpc_precip == 0).sum(dim='time')

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
    - ensemble corrected data
    - save ensemble corrected data to netcdf
    - metrics
    - save metrics to csv
    - multiplying factors
    - save multiplying factors as netcdf

    Returns:
    - Output available in folder output/{ensemble}/ corrected, metrics and factors
    """
    from dask import delayed
    from dask.distributed import Client
    from ensemble_methods import ensemble_method

    # Get the ensemble method used
    if not run_all_methods:
        if method is None:
            method = ensemble_method.__defaults__[0]
    else:
        method = None

    # Define the appropriate input and output directory paths
    input_dir = f'input'
    output_dir = f'output'
    corrected_path = f'{output_dir}/corrected'
    ensemble_path = f'{output_dir}/ensemble'
    
    # Create the output directories if they don't already exist
    os.makedirs(ensemble_path, exist_ok=True)
    os.makedirs(f'{ensemble_path}/corrected', exist_ok=True)
    os.makedirs(f'{ensemble_path}/metrics', exist_ok=True)
    os.makedirs(f'{ensemble_path}/factors', exist_ok=True)

    # Load the CPC data
    cpc_ds = xr.open_mfdataset(f'{input_dir}/cpc/*.nc')

    # Initialize a list to store the corrected datasets
    corrected_ds_list = []

    if run_all_methods:
        methods = ['scale', 'distribution', 'delta', 'lsc', 'lscdf', 'rcdfm', 'mlr', 
        'ann', 'kalman', 'bcsd', 'bcsdm', 'qq_mapping', 'eQM', 'aQM', 'gQM', 'gpdQM']
    else:
        methods = [method]

    for method in methods:
        method_path = f'{corrected_path}/{method}'
        corrected_ds_list.append(xr.open_mfdataset(f'{method_path}/*.nc'))

    # Use dask to schedule the computation of the ensemble corrected data
    client = Client()

    # Schedule the calculation of the ensemble corrected data
    ensemble_ds = delayed(ensemble_method)(corrected_ds_list, method=method)

    # Schedule the calculation of the statistical metrics for the ensemble corrected data
    ensemble_metrics = delayed(calculate_metrics)(ensemble_ds, cpc_ds)

    # Schedule the calculation of the multiplying factors for the ensemble corrected data
    ensemble_factors = delayed(create_multiplying_factors)(ensemble_ds)

    # Compute the ensemble corrected data, statistical metrics, and multiplying factors
    ensemble_ds = client.compute(ensemble_ds)
    ensemble_metrics = client.compute(ensemble_metrics)
    ensemble_factors = client.compute(ensemble_factors)

    # Encoding for CF 1.8
    cf18 = {'precipitation': {'dtype': 'float32', 'scale_factor': 0.1, 'zlib': True, \
    '_FillValue': -9999, 'add_offset': 0, 'least_significant_digit': 1}}

    # Save the ensemble corrected data, statistical metrics, and multiplying factors
    ensemble_ds.to_netcdf(f'{ensemble_path}/corrected/ensemble_corrected_{method}.nc', encoding=cf18)
    ensemble_metrics.to_csv(f'{ensemble_path}/metrics/ensemble_metrics_{method}.csv')
    for i, factor in enumerate(ensemble_factors):
        factor.to_netcdf(f'{ensemble_path}/factors/ensemble_multiplying_factor_{method}_{i+1}.nc', encoding=cf18)

if __name__ == '__main__':
    main(run_all_methods=True) # run all methods
    # main(method='decisiontree') # run decisiontree method
