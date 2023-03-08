# -*- coding: utf-8 -*-
"""
NAME
    uga_rainfallzone.py
DESCRIPTION
    Perform a focal linear regression between two sets of timeseries raster
REQUIREMENT
    It required os, pandas, geopandas, numpy, matplotlib, tqdm, joblib and sklearn module. 
    So it will work on any machine environment.
HOW-TO USE
    python uga_rainfallzone.py
NOTES
    The input required monthly precipitation data in csv which consist of 
    id, lon, lat, yyyymmdd, ..., yyyymmdd(n)
    the NoData (-9999.0) are manually cleaned before running this code
    Number of clusters assigned in the calculation, can be a specific integer or one of optimum result 
    generated by Calinski-Harahasz or Silhouette method
DATA
    This analysis use monthly timeseries precipitation 1981-2010 and 1990-2020 from CHIRPS in csv
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
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score, silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

# Choose the clustering method
# cluster_method = 'KMeans'
cluster_method = 'AgglomerativeClustering'

# Load the precipitation data
# precip_df = pd.read_csv("../csv/chirps_precip_1981_2010.csv", sep=";")
precip_df = pd.read_csv("../csv/chirps_precip_1991_2020.csv", sep=";")

# Rename the date columns for each dataframe
precip_df.rename(columns={'id': 'id', **{col: pd.to_datetime(col, format='%Y%m%d') for col in precip_df.columns[3:]}}, inplace=True)

# Melt the precipitation and temperature dataframes to a long format
precip_df = pd.melt(precip_df, id_vars=['id', 'lon', 'lat'], var_name='date', value_name='precipitation')

# Convert the date column to datetime type
precip_df['date'] = pd.to_datetime(precip_df['date'])

# Calculate the monthly mean precipitation for each location
precip_df["month"] = precip_df["date"].dt.month
monthly_precip_df = precip_df.groupby(["id", "lon", "lat", "month"], as_index=False).mean()
monthly_precip_df = monthly_precip_df.pivot_table(index=["id", "lon", "lat"], columns="month", values="precipitation").reset_index()
monthly_precip_df.columns.name = None
monthly_precip_df.columns = ["id", "lon", "lat", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

print("Compute the monthly mean precipitation completed")

# Compute the PCA 90
X = monthly_precip_df.drop(columns=["id", "lon", "lat"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(X_scaled)

print("Compute the PCA 90 completed")

def get_optimal_plot_calinski(X, cluster_method):
    """
    Compute the optimal number of clusters and plot the Calinski-Harabasz score as a function of the number of clusters.

    Parameters:
    - X: input data array
    - cluster_method: clustering algorithm to use, either "KMeans" or "AgglomerativeClustering"

    Returns:
    - optimal_k: optimal number of clusters
    """
    if cluster_method == "KMeans":
        model = KMeans
    elif cluster_method == "AgglomerativeClustering":
        model = AgglomerativeClustering
    else:
        raise ValueError("Invalid cluster_method type. Must be either 'KMeans' or 'AgglomerativeClustering'.")

    def compute_score(k):
        score_k = model(n_clusters=k)
        score_k.fit(X)
        score = calinski_harabasz_score(X, score_k.labels_)
        return score

    # Define the range of k values to explore
    k_values = range(2, 21)

    # Compute the Calinski-Harabasz score for each value of k
    scores = Parallel(n_jobs=-1)(delayed(compute_score)(k) for k in tqdm(k_values, desc="Calculating Calinski-Harabasz Scores"))

    # Find the index of the maximum score
    max_idx = np.argmax(scores)
    optimal_k = k_values[max_idx]

    # Plot the scores
    plt.plot(k_values, scores)
    plt.axvline(optimal_k, color='r', linestyle='--')
    plt.text(optimal_k+0.2, max(scores), f"Optimal k: {optimal_k}", color='r')
    plt.title(f"{cluster_method} - Calinski-Harabasz Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Calinski-Harabasz Score")
    plt.show()

    return optimal_k

def get_optimal_plot_silhouette(X, cluster_method):
    """
    Compute the optimal number of clusters for KMeans or AgglomerativeClustering using the Silhouette score,
    and plot the Silhouette score as a function of the number of clusters.

    Parameters:
    - X: input data array
    - cluster_method: clustering algorithm to use ("KMeans" or "AgglomerativeClustering")
    """
    def compute_score(k):
        if cluster_method == "KMeans":
            clusterer = KMeans(n_clusters=k)
        elif cluster_method == "AgglomerativeClustering":
            clusterer = AgglomerativeClustering(n_clusters=k)
        else:
            raise ValueError("Invalid cluster_method parameter. Must be 'KMeans' or 'AgglomerativeClustering'.")
        cluster_labels = clusterer.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        return silhouette_avg, sample_silhouette_values

    # Define the range of k values to explore
    k_values = range(2, 21)

    # Compute the Silhouette score for each value of k
    scores = Parallel(n_jobs=-1)(delayed(compute_score)(k) for k in tqdm(k_values, desc="Calculating Silhouette Scores"))

    # Extract the Silhouette score and sample Silhouette values for each k
    silhouette_scores, sample_silhouette_values = zip(*scores)

    # Find the index of the maximum score
    max_idx = np.argmax(silhouette_scores)
    max_k = k_values[max_idx]

    # Plot the scores
    plt.plot(k_values, silhouette_scores)
    plt.axvline(max_k, color='r', linestyle='--')
    plt.text(max_k+0.2, max(silhouette_scores), f"Optimal k: {max_k}", color='r')
    plt.title(f"{cluster_method} - Silhouette Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.show()

    # Return the optimal number of clusters
    return max_k

# Calculate the optimal number of clusters using various method
# Calinski-Harabasz
optimal_c = get_optimal_plot_calinski(X_pca, cluster_method)
# Silhouette
optimal_s = get_optimal_plot_silhouette(X_pca, cluster_method)

print("The optimal number of clusters using Calinski-Harabasz is: ", optimal_c)
print("The optimal number of clusters using Silhouette is: ", optimal_s)

def cluster_data(X_pca, cluster_method, n_clusters=None):
    """
    Cluster the input data using the specified method and number of clusters.

    Parameters:
    - X_pca: input data array
    - cluster_method: clustering method, either 'KMeans' or 'AgglomerativeClustering'
    - n_clusters: number of clusters, can be a specific integer or one of 'optimal_c' and 'optimal_s'

    Returns:
    - labels: array of cluster labels
    """
    if n_clusters == 'optimal_c':
        if cluster_method == 'KMeans':
            n_clusters = get_optimal_plot_calinski(X_pca, 'KMeans')
        elif mcluster_method == 'AgglomerativeClustering':
            n_clusters = get_optimal_plot_calinski(X_pca, 'AgglomerativeClustering')
        else:
            raise ValueError("Invalid clustering method. Choose 'KMeans' or 'AgglomerativeClustering'.")
    elif n_clusters == 'optimal_s':
        if cluster_method == 'KMeans':
            n_clusters = get_optimal_plot_silhouette(X_pca, 'KMeans')
        elif cluster_method == 'AgglomerativeClustering':
            n_clusters = get_optimal_plot_silhouette(X_pca, 'AgglomerativeClustering')
        else:
            raise ValueError("Invalid clustering method. Choose 'KMeans' or 'AgglomerativeClustering'.")
    elif isinstance(n_clusters, int):
        pass
    else:
        raise ValueError("Invalid value for n_clusters parameter.")
    
    if cluster_method == 'KMeans':
        cluster_model = KMeans(n_clusters=n_clusters)
    elif cluster_method == 'AgglomerativeClustering':
        cluster_model = AgglomerativeClustering(n_clusters=n_clusters)

    pbar = tqdm(total=1, desc=f"Performing {cluster_method} Clustering")
    cluster_model.fit(X_pca)
    pbar.update(1)
    
    labels = cluster_model.labels_

    return labels, n_clusters

# Other ways to assign a single cluster to each row based on the 360 monthly timeseries 
# of precipitation and temperature data. One approach is to calculate the distance 
# between each row and the centroids of the clusters obtained from K-Means clustering. 
# Then, assign the closest cluster to each row as its climate zone.

# Cluster the data using the defined cluster_method
labels, n_clusters = cluster_data(X_pca, cluster_method, n_clusters=14)

# Calculate the centroids for each cluster
centroids = np.zeros((n_clusters, X_pca.shape[1]))
for i in range(n_clusters):
    mask = (labels == i)
    centroids[i,:] = np.mean(X_pca[mask,:], axis=0)

# Calculate the Euclidean distance between each row and the centroids
distances = np.sqrt(np.sum(np.square(X_pca[:, None] - centroids), axis=2))

# In this code, we use the np.argmin() function to find the index of the closest centroid 
# to each row. This approach preserves the details of climate characteristics captured by 
# the monthly timeseries data.
# Assign the closest cluster to each row
monthly_precip_df["climate_zone"] = np.argmin(distances, axis=1)

# Group the merged dataframe by id, lon, and lat and take the mode of the climate zone for each group
monthly_precip_df_centroid = monthly_precip_df.groupby(["id", "lon", "lat"])["climate_zone"].apply(lambda x: x.mode()[0]).reset_index()

print("Group the merged dataframe completed")

def plot_climate_zone_map(climate_zone_csv, shapefile_path):
    """
    Plot the climate zones as a point map.

    Parameters:
    - climate_zone_csv: path to the CSV file with lon, lat, and climate_zone columns
    """
    # Load the data from the CSV file
    df = pd.read_csv(climate_zone_csv)

    # Extract the lon, lat, and climate_zone columns
    lon = df["lon"]
    lat = df["lat"]
    climate_zone = df["climate_zone"]

    # Create a scatter plot of the climate zones
    plt.figure(figsize=(12, 10))
    plt.scatter(lon, lat, c=climate_zone, cmap="tab20")
    
    # Load the polygon shapefile using geopandas
    gdf = gpd.read_file(shapefile_path)

    # Plot the polygon shapefile
    gdf.plot(ax=plt.gca(), facecolor="None", edgecolor="black", linewidth=1)

    # Add a colorbar
    cbar = plt.colorbar()
    cbar.set_label("New Rainfall Zone and the 1995 Climatological Rainfall Zone")

    # Set the x and y axis labels
    # You need to adjust the title with the year of data information if needed
    plt.title("Rainfall Zone based on monthly CHIRPS\nAlgorithm: "f"{cluster_method}", fontsize=16, fontweight='bold', ha='center')
    plt.text(0.5, -0.15, "The existing climatological rainfall zone based on study from C.P.K.Basalirwa in 1995\nusing monthly record from 102 rain-gauge stations for the years 1940-1975\nhttps://doi.org/10.1002/joc.3370151008", fontsize=12, fontweight='normal', ha='center', transform=plt.gca().transAxes)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    # Show the plot
    plt.show()

# Save the grouped dataframe to a new CSV file
output_df_centroid = monthly_precip_df_centroid[['id', 'lon', 'lat', 'climate_zone']]
output_df_centroid.to_csv("../csv/uga_climatezone_eucl_{0}_{1}_p_ai.csv".format(n_clusters, cluster_method), index=False)
print("Save the output to csv completed")

# Plot map
plot_climate_zone_map("../csv/uga_climatezone_eucl_{0}_{1}_p_ai.csv".format(n_clusters, cluster_method))

print("All the process completed")
