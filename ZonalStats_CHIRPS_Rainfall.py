import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.transform import from_bounds
from rasterstats import zonal_stats
import os
from datetime import datetime


def create_output_directory():
    """Create a directory for storing outputs with timestamp"""
    base_dir = 'ncfiles/spi7/zonal_statistics_output'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'analysis_{timestamp}')
    os.makedirs(output_dir)

    return output_dir


def calculate_precipitation_zscores(nc_file, shp_file, start_year=2018):
    """
    Calculate z-scores and zonal statistics for precipitation data from 2018 onwards.
    """
    # Create output directory
    output_dir = create_output_directory()
    output_csv = os.path.join(output_dir, 'zonal_stats_zscore_2018onwards.csv')

    # Load shapefile
    shp_df = gpd.read_file(shp_file)

    # Verify ADM3_PCODE exists
    if 'ADM3_PCODE' not in shp_df.columns:
        raise KeyError("ADM3_PCODE field is not found in the shapefile.")

    # Load NetCDF data
    print(f"Loading NetCDF file: {nc_file}")
    nc_ds = xr.open_dataset(nc_file)
    precip_var = list(nc_ds.data_vars)[0]  # Get first variable
    print(f"Variable found: {precip_var}")

    # Print time range information
    time_range = pd.to_datetime(nc_ds.time.values)
    print(f"Full dataset time range: {time_range.min()} to {time_range.max()}")

    # Select data from 2018 onwards
    nc_var = nc_ds[precip_var].sel(time=slice(f"{start_year}-01-01", None))

    # Verify we have data after filtering
    times = pd.to_datetime(nc_var.time.values)
    if len(times) == 0:
        raise ValueError(
            f"No data found for period from {start_year} onwards. Please check the date range in your NetCDF file.")

    print(f"Processing {len(times)} months from {start_year} onwards...")
    print(f"Selected time range: {times.min()} to {times.max()}")

    # Extract coordinate information - handle both lat/lon and latitude/longitude names
    lat_values = nc_ds['latitude'].values if 'latitude' in nc_ds else nc_ds['lat'].values
    lon_values = nc_ds['longitude'].values if 'longitude' in nc_ds else nc_ds['lon'].values

    # Compute spatial bounds
    lat_min, lat_max = lat_values.min(), lat_values.max()
    lon_min, lon_max = lon_values.min(), lon_values.max()

    # Initialize results
    results = []

    # Calculate monthly climatology (mean and std) using all available data
    print("Calculating monthly climatology...")
    monthly_mean = nc_ds[precip_var].groupby('time.month').mean('time')
    monthly_std = nc_ds[precip_var].groupby('time.month').std('time')

    # Process each timestep from 2018 onwards
    total_times = len(times)

    for i, time in enumerate(times, 1):
        year = time.year
        month = time.month
        print(f"Processing: {time.strftime('%Y-%m')} ({i}/{total_times})")

        # Get the data for this time
        time_data = nc_var.sel(time=time)

        # Get climatology for this month
        month_mean = monthly_mean.sel(month=month)
        month_std = monthly_std.sel(month=month)

        # Calculate z-score
        with np.errstate(divide='ignore', invalid='ignore'):  # Handle division by zero
            zscore = (time_data - month_mean) / month_std
            zscore = zscore.fillna(0)  # Replace NaN values with 0

        # Create transform for this slice
        transform = from_bounds(
            lon_min, lat_min, lon_max, lat_max,
            zscore.shape[1], zscore.shape[0]
        )

        # Calculate zonal statistics
        stats = zonal_stats(
            shp_df,
            zscore.values,
            affine=transform,
            stats=["mean", "min", "max", "std", "count"]
        )

        # Add metadata to results
        for j, stat in enumerate(stats):
            stat['year'] = year
            stat['month'] = month
            stat['adm3_pcode'] = shp_df.loc[j, 'ADM3_PCODE']

            # Add administrative names if available
            for col in ['ADM3_EN', 'ADM2_EN', 'ADM1_EN']:
                if col in shp_df.columns:
                    stat[col.lower()] = shp_df.loc[j, col]

            results.append(stat)

    # Convert to DataFrame
    print("Creating final DataFrame...")
    df = pd.DataFrame(results)

    if len(df) == 0:
        raise ValueError("No results generated. Please check your input data.")

    # Add interpretation columns
    df['zscore_category'] = pd.cut(
        df['mean'].fillna(0),  # Handle any NaN values
        bins=[-np.inf, -2, -1, 1, 2, np.inf],
        labels=['Very Dry', 'Moderately Dry', 'Normal', 'Moderately Wet', 'Very Wet']
    )

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"Analysis complete. Results saved to directory: {output_dir}")
    print(f"Z-score zonal statistics saved as: {output_csv}")

    # Save processing information
    info_file = os.path.join(output_dir, 'processing_info.txt')
    with open(info_file, 'w') as f:
        f.write(f"Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input NetCDF file: {nc_file}\n")
        f.write(f"Input shapefile: {shp_file}\n")
        f.write(f"Time period processed: {times.min()} to {times.max()}\n")
        f.write(f"Number of administrative units: {len(shp_df)}\n")
        f.write(f"Climatology period: {time_range.min()} to {time_range.max()}\n")

    return df, output_dir


def plot_zscore_map(df, shp_df, year, month, output_dir):
    """
    Create a choropleth map of z-scores for a specific year and month

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the zonal statistics results
    shp_df : geopandas.GeoDataFrame
        GeoDataFrame containing the shapefile data
    year : int
        Year to plot
    month : int
        Month to plot
    output_dir : str
        Directory to save the output map
    """
    import matplotlib.pyplot as plt

    output_file = os.path.join(output_dir, f'zscore_map_{year}_{month:02d}.png')

    # Merge data with shapefile
    month_data = df[(df['year'] == year) & (df['month'] == month)]
    merged = shp_df.merge(month_data, how='left', left_on='ADM3_PCODE', right_on='adm3_pcode')

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    # Plot the data
    merged.plot(
        column='mean',
        cmap='RdBu_r',
        legend=True,
        legend_kwds={
            'label': 'Precipitation Z-score',
            'orientation': 'vertical',
            'shrink': 0.5
        },
        missing_kwds={'color': 'lightgrey'},
        ax=ax
    )

    # Add title
    month_name = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                  5: 'May', 6: 'June', 7: 'July', 8: 'August',
                  9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    plt.title(f'Precipitation Z-scores - {month_name[month]} {year}', pad=20, fontsize=14)

    # Remove axes
    plt.axis('off')

    # Add colorbar labels
    ax = plt.gca()
    if len(ax.get_children()) > 1:  # Check if colorbar exists
        cbar = ax.get_children()[-2]  # Get colorbar
        cbar.ax.text(3.5, -2, 'Drier', ha='center', va='top')
        cbar.ax.text(3.5, 2, 'Wetter', ha='center', va='bottom')

    # Save the plot
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Map saved as: {output_file}")

# Example usage
if __name__ == "__main__":
    nc_file = 'ncfiles/ethiopia_CHIRPS_precipitation_post_2018.nc'
    shp_file = 'eth_adm_csa_bofedb_2021_shp/eth_admbnda_adm3_csa_bofedb_2021.shp'

    try:
        # Calculate z-scores and zonal statistics
        results_df, output_dir = calculate_precipitation_zscores(nc_file, shp_file, start_year=2018)

        # Create map for most recent month if results exist
        if len(results_df) > 0:
            latest_data = results_df.sort_values(['year', 'month']).iloc[-1]
            shp_df = gpd.read_file(shp_file)
            plot_zscore_map(results_df, shp_df, latest_data['year'], latest_data['month'], output_dir)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check your input files and make sure they contain the expected data.")