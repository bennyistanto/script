# -*- coding: utf-8 -*-
"""
NAME
    taylor_diagram_outmetrics.py
    Generate Taylor Diagram from all metrics output
DESCRIPTION
    Input data for this script will use metric.csv generated by imerg_cpc_lscdf01.py.
REQUIREMENT
    It required numpy, pandas, metpy, matplotlib and xarray module. So it will work on any machine environment
EXAMPLES
    python taylor_diagram_outmetrics.py
NOTES
    To visualize the metrics output using a Taylor diagram, you can use the taylor_diagram function 
    from the taylor_diagram module of the metpy library. This function allows you to plot the relative 
    bias, Pearson correlation coefficient, and root mean square error of a dataset on a Taylor diagram, 
    with reference to a reference dataset.
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
import pandas as pd
import numpy as np
from metpy.plots import taylor_diagram
import matplotlib.pyplot as plt

# Set the input and output folder locations
input_folder = '/path/to/input/folder/'
output_folder = '/path/to/output/folder/'

# Load the metrics output from the bias correction
metrics = pd.read_csv(f'{input_folder}metrics.csv')

# Select the reference dataset (CPC in this example) and the corrected datasets (IMERG LS and LSCDF)
ref = metrics[metrics['dataset'] == 'cpc']
imerg_ls = metrics[metrics['dataset'] == 'imerg_ls']
imerg_lscdf = metrics[metrics['dataset'] == 'imerg_lscdf']

# Extract the relative bias, Pearson correlation coefficient, and root mean square error for each dataset
ref_r, ref_p, ref_rmse = ref['relative_bias'], ref['pearson'], ref['rmse']
imerg_ls_r, imerg_ls_p, imerg_ls_rmse = imerg_ls['relative_bias'], imerg_ls['pearson'], imerg_ls['rmse']
imerg_lscdf_r, imerg_lscdf_p, imerg_lscdf_rmse = imerg_lscdf['relative_bias'], imerg_lscdf['pearson'], imerg_lscdf['rmse']

# Create the Taylor diagram figure
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='polar')

# Add the reference dataset to the Taylor diagram
taylor_diagram(ref_rmse, ref_r, ref_p, marker='o', color='k', markersize=10, label='CPC')

# Add the IMERG LS dataset to the Taylor diagram
taylor_diagram(imerg_ls_rmse, imerg_ls_r, imerg_ls_p, marker='s', color='b', markersize=7, label='IMERG LS')

# Add the IMERG LSCDF dataset to the Taylor diagram
taylor_diagram(imerg_lscdf_rmse, imerg_lscdf_r, imerg_lscdf_p, marker='^', color='r', markersize=7, label='IMERG LSCDF')

# Add the gridlines and legend to the Taylor diagram
plt.grid(True)
ax.legend(loc='upper left')

# Save the Taylor diagram to a PNG file in the output folder
plt.savefig(f'{output_folder}taylor_diagram.png', dpi=300, bbox_inches='tight')

# Show the Taylor diagram
plt.show()