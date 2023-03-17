# -*- coding: utf-8 -*-
"""
NAME
    precip_improvement_plots.py
    Generate plots on improvements daily rainfall after the bias correction
DESCRIPTION
    Input data for this script will use the gauge and satellite-based precipitation  
    estimates data, and corrected precipitation from bias-correction process.
REQUIREMENT
    It required numpy, pandas, scipy, matplotlib, seaborn and xarray module. 
    So it will work on any machine environment
EXAMPLES
    python precip_improvement_plots.py
NOTES
    The code imports the necessary libraries, loads the reference data (CPC) and the 
    original satellite data (IMERG), and calculates the daily bias and RMSE. The 
    performance_metrics() function computes bias, MAE, and RMSE before and after the 
    correction. The script then defines plotting functions for scatter plots, PDFs, CDFs, 
    Bland-Altman plots, and performance metrics. Finally, for each method and year, 
    the script reads the corrected precipitation data, creates and saves the plots, and 
    prints the performance metrics. Make sure the paths and filenames for your input and 
    output files match your file structure.
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
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Define the list of methods
methods = ['scale', 'distribution', 'delta', 'lsc', 'lscdf', 'rcdfm', 'mlr', 
'ann', 'kalman', 'bcsd', 'bcsdm', 'qq_mapping', 'eQM', 'aQM', 'gQM', 'gpdQM']  # same as before

# Define the input and output directory paths
input_dir = f'input'
imerg_path = f'{input_dir}/imerg'
cpc_path = f'{input_dir}/cpc'
output_dir = f'output'
plots_dir = f'{output_dir}/plots'

# Create the output directories if they don't already exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Load reference gauge data (assumed to be a NetCDF file)
cpc_ds = xr.open_dataset(f'{cpc_path}/*.nc')['precip']

# Load original satellite data (assumed to be a NetCDF file)
imerg_ds = xr.open_dataset(f'{imerg_path}/*.nc')['precipitationCal']

def calculate_daily_bias(cpc_ds, data):
    """
    This function calculates the daily bias between the corrected data and the reference data (CPC) 
    by taking the difference and then finding the mean over the latitude and longitude dimensions.

    Parameters:
    - cpc_ds: Reference gauge dataset (CPC)
    - data: Corrected dataset

    Returns: Daily bias value
    """
    return (data - cpc_ds).mean(dim=['lat', 'lon'])

def calculate_daily_rmse(cpc_ds, data):
    """
    This function calculates the daily root mean square error (RMSE) between the corrected data and 
    the reference data (CPC) by taking the squared difference, finding the mean over the latitude and 
    longitude dimensions, and then taking the square root.

    Parameters:
    - cpc_ds: Reference gauge dataset (CPC)
    - data: Corrected dataset

    Returns: Daily RMSE value
    """
    return np.sqrt(((data - cpc_ds) ** 2).mean(dim=['lat', 'lon']))

def performance_metrics(cpc_ds, imerg_ds, corrected_ds):
    """
    This function calculates the bias, mean absolute error (MAE), and RMSE before and after the bias 
    correction. It then returns a DataFrame containing these metrics.

    Parameters:
    - cpc_ds: Reference gauge dataset (CPC)
    - imerg_ds: Original satellite dataset (IMERG)
    - corrected_ds: Corrected dataset

    Returns: DataFrame with the performance metrics before and after the correction
    """
    bias_before = np.mean(imerg_ds - cpc_ds)
    bias_after = np.mean(corrected_ds - cpc_ds)
    
    mae_before = np.mean(np.abs(imerg_ds - cpc_ds))
    mae_after = np.mean(np.abs(corrected_ds - cpc_ds))
    
    rmse_before = np.sqrt(np.mean((imerg_ds - cpc_ds) ** 2))
    rmse_after = np.sqrt(np.mean((corrected_ds - cpc_ds) ** 2))
    
    return pd.DataFrame({
        'Metric': ['Bias', 'MAE', 'RMSE'],
        'Before Correction': [bias_before, mae_before, rmse_before],
        'After Correction': [bias_after, mae_after, rmse_after]
    })

def plot_scatter(ax, cpc_ds, data, label, color):
    """
    This function creates a scatter plot comparing the reference data to the original or corrected data, 
    with the reference data on the x-axis and the data on the y-axis.

    Parameters:
    - ax: Matplotlib axis object for plotting
    - cpc_ds: Reference gauge dataset (CPC)
    - data: Dataset to be compared (either original or corrected)
    - label: Label for the plotted data
    - color: Color for the plotted data points
    """
    ax.scatter(cpc_ds, data, alpha=0.5, color=color, label=label)
    ax.set_xlabel('Reference Data')
    ax.set_ylabel('Data')
    ax.set_title('Scatter Plot')

def plot_pdf(ax, data, label, color):
    """
    This function creates a probability density function (PDF) plot for the reference data, 
    the original satellite data, or the corrected data. It estimates the PDF using Gaussian 
    kernel density estimation and then plots the estimated PDF.

    Parameters:
    - ax: Matplotlib axis object for plotting
    - data: Dataset for which the PDF will be plotted
    - label: Label for the plotted data
    - color: Color for the plotted data points
    """
    kernel = gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 1000)
    y_vals = kernel(x_vals)
    ax.plot(x_vals, y_vals, color=color, label=label)
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.set_title('PDF')

def plot_cdf(ax, data, label, color):
    """
    This function creates a cumulative distribution function (CDF) plot for the reference data, 
    the original satellite data, or the corrected data.

    Parameters:
    - ax: Matplotlib axis object for plotting
    - data: Dataset for which the CDF will be plotted
    - label: Label for the plotted data
    - color: Color for the plotted data points
    """
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, cdf, color=color, label=label)
    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('CDF')

def plot_bland_altman(ax, cpc_ds, data, label, color):
    """
    This function creates a Bland-Altman plot, which is a scatter plot of the differences between 
    the reference data and the original or corrected data against the average of the reference data 
    and the data. It also plots the mean difference and the 95% limits of agreement.

    Parameters:
    - ax: Matplotlib axis object for plotting
    - cpc_ds: Reference gauge dataset (CPC)
    - data: Dataset to be compared (either original or corrected)
    - label: Label for the plotted data
    - color: Color for the plotted data points
    """
    mean_difference = np.mean(cpc_ds - data)
    std_difference = np.std(cpc_ds - data, ddof=1)
    ax.scatter((cpc_ds + data) / 2, cpc_ds - data, alpha=0.5, color=color, label=label)
    ax.axhline(mean_difference, color='k', linestyle='--')
    ax.axhline(mean_difference + 1.96 * std_difference, color='k', linestyle='--')
    ax.axhline(mean_difference - 1.96 * std_difference, color='k', linestyle='--')
    ax.set_xlabel('Mean of Reference Data and Data')
    ax.set_ylabel('Difference between Reference Data and Data')
    ax.set_title('Bland-Altman Plot')

def plot_performance_metrics(ax, metrics_df):
    """
    This function creates a bar plot of the performance metrics (bias, MAE, and RMSE) 
    before and after the correction.

    Parameters:
    - ax: Matplotlib axis object for plotting
    - metrics_df: DataFrame with the performance metrics before and after the correction
    """
    metrics_df.plot.bar(x='Metric', y=['Before Correction', 'After Correction'], ax=ax)
    ax.set_ylabel('Value')
    ax.set_title('Performance Metrics')
    ax.grid(True)
    
for method in methods:
    for year in range(2001, 2023):
        # Load the corrected precipitation data for the method (assumed to be a NetCDF file)
        corrected_ds = xr.open_dataset(f'output/{method}/corrected/corrected_{method}_{year}.nc')['precipitation']

        # Get the data for this year
        imerg_year_ds = imerg_ds.sel(time=imerg_ds['time.year'] == year)
        cpc_year_ds = cpc_ds.sel(time=cpc_ds['time.year'] == year)

        # Scatter plot
        fig, scatter_ax = plt.subplots(figsize=(6, 6))
        plot_scatter(scatter_ax, cpc_ds, imerg_ds, label='Original Satellite', color='r')
        plot_scatter(scatter_ax, cpc_ds, corrected_ds, label=f'{method} Corrected', color='b')
        scatter_ax.legend(loc='upper left')
        scatter_ax.grid(True)
        fig.savefig(f'{plots_dir}/{method}_scatter_plot.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # PDF plot
        fig, pdf_ax = plt.subplots(figsize=(6, 6))
        plot_pdf(pdf_ax, cpc_ds, label='Reference', color='k')
        plot_pdf(pdf_ax, imerg_ds, label='Original Satellite', color='r')
        plot_pdf(pdf_ax, corrected_ds, label=f'{method} Corrected', color='b')
        pdf_ax.legend(loc='upper right')
        pdf_ax.grid(True)
        fig.savefig(f'{plots_dir}/{method}_pdf_plot.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # CDF plot
        fig, cdf_ax = plt.subplots(figsize=(6, 6))
        plot_cdf(cdf_ax, cpc_ds, label='Reference', color='k')
        plot_cdf(cdf_ax, imerg_ds, label='Original Satellite', color='r')
        plot_cdf(cdf_ax, corrected_ds, label=f'{method} Corrected', color='b')
        cdf_ax.legend(loc='upper left')
        cdf_ax.grid(True)
        fig.savefig(f'{plots_dir}/{method}_cdf_plot.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Bland-Altman plot
        fig, bland_altman_ax = plt.subplots(figsize=(6, 6))
        plot_bland_altman(bland_altman_ax, cpc_ds, imerg_ds, label='Original Satellite', color='r')
        plot_bland_altman(bland_altman_ax, cpc_ds, corrected_ds, label=f'{method} Corrected', color='b')
        bland_altman_ax.legend(loc='upper left')
        bland_altman_ax.grid(True)
        fig.savefig(f'{plots_dir}/{method}_bland_altman_plot.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Performance metrics
        metrics_df = performance_metrics(cpc_ds, imerg_ds, corrected_ds)
        print(f"Performance Metrics for {method}:")
        print(metrics_df.to_string(index=False))
        print("\n")

        fig, metrics_ax = plt.subplots(figsize=(6, 6))
        plot_performance_metrics(metrics_ax, metrics_df)
        fig.savefig(f'{plots_dir}/{method}_performance_metrics_plot.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        cpc_ds.close()
        imerg_ds.close()
        corrected_ds.close()
