import xarray as xr
import numpy as np
import pandas as pd
from climate_indices import indices
from climate_indices.compute import Periodicity
from climate_indices.indices import Distribution
import os


def process_chirps_data(input_file):
    """
    Process CHIRPS data for Ethiopia region, limiting to 1981-2023
    """
    # Check if file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"CHIRPS data file not found at: {input_file}")

    print("Loading CHIRPS data...")
    ds = xr.open_dataset(input_file)

    # Print initial data information
    print("\nInitial data information:")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Coordinates: {list(ds.coords)}")

    # CHIRPS typically uses different coordinate names
    ds = ds.rename({
        'time': 'time',  # Usually already correct in CHIRPS
        'latitude': 'lat' if 'latitude' in ds.coords else 'lat',
        'longitude': 'lon' if 'longitude' in ds.coords else 'lon'
    })

    # Subset time to 1981-2023
    print("\nSubsetting time period to 1981-2023...")
    ds = ds.sel(time=slice('1981-01-01', '2023-12-31'))

    # Define Ethiopia's bounds
    lat_min, lat_max = 3.5, 14.8
    lon_min, lon_max = 33.0, 47.8

    # Subset data for Ethiopia
    print("\nSubsetting data for Ethiopia region...")
    ds_ethiopia = ds.sel(
        lat=slice(lat_min, lat_max),  # Note: CHIRPS might have different lat orientation
        lon=slice(lon_min, lon_max)
    )

    # CHIRPS precipitation variable is typically named 'precip'
    if 'precip' in ds_ethiopia:
        ds_ethiopia = ds_ethiopia.rename({'precip': 'precipitation'})

    print("\nProcessed data dimensions:")
    print(f"Time steps: {len(ds_ethiopia.time)}")
    print(f"Latitude points: {len(ds_ethiopia.lat)}")
    print(f"Longitude points: {len(ds_ethiopia.lon)}")
    print(f"Time range: {ds_ethiopia.time.values[0]} to {ds_ethiopia.time.values[-1]}")

    # Check for missing values
    if np.any(np.isnan(ds_ethiopia['precipitation'])):
        print("\nWarning: Dataset contains missing values!")
        print("Number of missing values:", np.isnan(ds_ethiopia['precipitation'].values).sum())

    return ds_ethiopia


def calculate_spi7(ds):
    """
    Calculate 7-month SPI from processed CHIRPS data
    """
    print("\nPreparing data for SPI-7 calculation...")

    # Get precipitation data - CHIRPS data is typically already in mm
    precip = ds['precipitation'].values

    # Basic data validation
    if np.any(precip < 0):
        raise ValueError("Negative precipitation values found in the dataset")

    # Reshape to 2D (time, space) as required by climate_indices
    original_shape = precip.shape
    time_steps = original_shape[0]
    space_points = original_shape[1] * original_shape[2]
    precip_2d = precip.reshape(time_steps, space_points)

    print(f"Data shape for SPI calculation: {precip_2d.shape}")

    # Set up SPI parameters
    params = {
        'scale': 7,  # 7-month SPI
        'data_start_year': 1981,
        'calibration_year_initial': 1981,
        'calibration_year_final': 2023,
        'periodicity': Periodicity.monthly,
        'distribution': Distribution.gamma
    }

    print("\nCalculating SPI-7 with parameters:")
    print(f"Time period: {params['data_start_year']}-{params['calibration_year_final']}")
    print(f"Scale: 7 months")

    # Calculate SPI
    spi_values = indices.spi(
        values=precip_2d,
        scale=params['scale'],
        data_start_year=params['data_start_year'],
        calibration_year_initial=params['calibration_year_initial'],
        calibration_year_final=params['calibration_year_final'],
        periodicity=params['periodicity'],
        distribution=params['distribution']
    )

    # Reshape back to original dimensions
    spi_3d = np.array(spi_values).reshape(original_shape)

    # Add SPI to dataset
    ds['spi7'] = (('time', 'lat', 'lon'), spi_3d)
    ds.spi7.attrs['long_name'] = 'Standardized Precipitation Index (7-month)'
    ds.spi7.attrs['units'] = 'standardized units'
    ds.spi7.attrs['calibration_period'] = '1981-2023'

    return ds


def main():
    try:
        input_file = r"C:\Users\olanrewaju\Downloads\chirps-v2.0.monthly.nc"
        output_file = 'ncfiles/spi7/ethiopia_chirps_spi7_1981_2023.nc'

        # Process CHIRPS data
        ds_processed = process_chirps_data(input_file)

        # Calculate SPI-7
        print("\nCalculating SPI-7...")
        ds_with_spi = calculate_spi7(ds_processed)

        # Save results
        print(f"\nSaving results to {output_file}")
        encoding = {
            'spi7': {'zlib': True, 'complevel': 4, '_FillValue': np.nan},
            'precipitation': {'zlib': True, 'complevel': 4, '_FillValue': np.nan}
        }
        ds_with_spi.to_netcdf(output_file, encoding=encoding)

        print("\nProcessing completed successfully!")

    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()