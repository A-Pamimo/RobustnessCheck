import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.transform import from_bounds
from rasterstats import zonal_stats
import os
from datetime import datetime


def convert_longitude_0_360_to_180(lon):
    """Convert longitude from 0-360 to -180 to 180 format"""
    return ((lon + 180) % 360) - 180


def create_output_directory():
    """Create a directory for storing outputs with timestamp"""
    base_dir = 'ncfiles/gdd/zonal_statistics_output'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'analysis_{timestamp}')
    os.makedirs(output_dir)

    return output_dir


def calculate_gdd(temp_data, base_temp=10.0, upper_thresh=35.0):
    """Calculate Growing Degree Days"""
    # Convert temperature from Kelvin to Celsius
    temp_celsius = temp_data - 273.15

    # Apply upper threshold
    temp_celsius = temp_celsius.where(temp_celsius <= upper_thresh, upper_thresh)

    # Calculate GDD
    gdd = temp_celsius - base_temp
    gdd = gdd.where(gdd > 0, 0)

    return gdd


def calculate_gdd_zonal_stats(nc_file, shp_file, start_year=2018):
    """Calculate GDD and zonal statistics for temperature data."""
    output_dir = create_output_directory()
    output_csv = os.path.join(output_dir, 'zonal_stats_gdd_2018onwards.csv')

    # Load shapefile
    shp_df = gpd.read_file(shp_file)
    if 'ADM3_PCODE' not in shp_df.columns:
        raise KeyError("ADM3_PCODE field is not found in the shapefile.")

    # Load NetCDF data
    print(f"Loading NetCDF file: {nc_file}")
    nc_ds = xr.open_dataset(nc_file)

    # Print dataset information
    print("\nDataset dimensions:")
    print(nc_ds.dims)

    # Get temperature variable (t2m for ERA5)
    temp_var = 't2m'
    print(f"\nVariable found: {temp_var}")

    # Define Ethiopia's bounding box
    ethiopia_bounds = {
        'lat_min': 3.5,
        'lat_max': 14.8,
        'lon_min': 33.0,
        'lon_max': 47.5
    }

    # Print coordinate ranges
    print("\nInitial coordinate ranges:")
    print(f"Latitude: {nc_ds.latitude.values.min():.2f} to {nc_ds.latitude.values.max():.2f}")
    print(f"Longitude before conversion: {nc_ds.longitude.values.min():.2f} to {nc_ds.longitude.values.max():.2f}")

    # Convert longitude coordinates
    nc_ds.coords['longitude'] = convert_longitude_0_360_to_180(nc_ds.longitude)
    nc_ds = nc_ds.sortby('longitude')

    # Select Ethiopia region
    if nc_ds.latitude[0] > nc_ds.latitude[-1]:
        nc_ds = nc_ds.sel(
            latitude=slice(ethiopia_bounds['lat_max'], ethiopia_bounds['lat_min']),
            longitude=slice(ethiopia_bounds['lon_min'], ethiopia_bounds['lon_max'])
        )
    else:
        nc_ds = nc_ds.sel(
            latitude=slice(ethiopia_bounds['lat_min'], ethiopia_bounds['lat_max']),
            longitude=slice(ethiopia_bounds['lon_min'], ethiopia_bounds['lon_max'])
        )

    # Select data from start_year onwards
    nc_var = nc_ds[temp_var].sel(valid_time=slice(f"{start_year}-01-01", None))

    # Verify data availability
    times = pd.to_datetime(nc_var.valid_time.values)
    if len(times) == 0:
        raise ValueError(f"No data found for period from {start_year} onwards.")

    print(f"\nProcessing {len(times)} time steps from {start_year} onwards...")
    print(f"Selected time range: {times.min()} to {times.max()}")

    # Extract coordinates
    lat_values = nc_ds.latitude.values
    lon_values = nc_ds.longitude.values
    lat_min, lat_max = lat_values.min(), lat_values.max()
    lon_min, lon_max = lon_values.min(), lon_values.max()

    results = []
    total_times = len(times)

    for i, time in enumerate(times, 1):
        print(f"Processing: {time.strftime('%Y-%m')} ({i}/{total_times})")

        # Get temperature data for this time
        time_data = nc_var.sel(valid_time=time)

        # Calculate GDD
        gdd = calculate_gdd(time_data)

        # Create transform
        transform = from_bounds(
            lon_min, lat_min, lon_max, lat_max,
            gdd.shape[1], gdd.shape[0]
        )

        # Calculate zonal statistics
        stats = zonal_stats(
            shp_df,
            gdd.values,
            affine=transform,
            stats=["mean", "min", "max", "sum", "count"]
        )

        # Add metadata
        for j, stat in enumerate(stats):
            stat['year'] = time.year
            stat['month'] = time.month
            stat['adm3_pcode'] = shp_df.loc[j, 'ADM3_PCODE']

            # Add administrative names if available
            for col in ['ADM3_EN', 'ADM2_EN', 'ADM1_EN']:
                if col in shp_df.columns:
                    stat[col.lower()] = shp_df.loc[j, col]

            results.append(stat)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    if len(df) == 0:
        raise ValueError("No results generated. Please check your input data.")

    # Add GDD categories
    df['gdd_category'] = pd.cut(
        df['sum'],
        bins=[-np.inf, 500, 1000, 1500, 2000, np.inf],
        labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    )

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"\nAnalysis complete. Results saved to: {output_csv}")

    # Save processing information
    info_file = os.path.join(output_dir, 'processing_info.txt')
    with open(info_file, 'w') as f:
        f.write(f"Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input NetCDF file: {nc_file}\n")
        f.write(f"Input shapefile: {shp_file}\n")
        f.write(f"Time period processed: {times.min()} to {times.max()}\n")
        f.write(f"Number of administrative units: {len(shp_df)}\n")
        f.write(f"Base temperature: 10°C\n")
        f.write(f"Upper threshold: 35°C\n")
        f.write(f"Spatial extent: Lat [{lat_values.min():.2f}, {lat_values.max():.2f}], "
                f"Lon [{lon_values.min():.2f}, {lon_values.max():.2f}]\n")

    return df, output_dir


# Example usage
if __name__ == "__main__":
    nc_file = 'ncfiles/2m temp ERA5.nc'  # Your ERA5 temperature file
    shp_file = 'eth_adm_csa_bofedb_2021_shp/eth_admbnda_adm3_csa_bofedb_2021.shp'

    try:
        results_df, output_dir = calculate_gdd_zonal_stats(nc_file, shp_file, start_year=2018)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Please check your input files and make sure they contain the expected data.")