import xarray as xr
import numpy as np
from climate_indices import indices
from climate_indices.compute import Periodicity
from climate_indices.indices import Distribution
import os


def load_and_combine_era5_data(input_files):
  """
  Load and combine multiple ERA5 data files, subsetting data for Ethiopia and 1981-2023
  """
  print("Loading and combining ERA5 data files...")
  datasets = []

  # Define Ethiopia's bounds
  lat_min, lat_max = 3.5, 14.8
  lon_min, lon_max = 33.0, 47.8
  time_range = slice('1981-01-01', '2023-12-31')

  for file in input_files:
    print(f"Loading file: {file}")
    ds = xr.open_dataset(file)

    # Rename coordinates to standard names
    ds = ds.rename({
        'valid_time': 'time',
        'latitude': 'lat',
        'longitude': 'lon'
    })

    # Subset data for Ethiopia and time range before appending
    ds_subset = ds.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max), time=time_range)
    datasets.append(ds_subset)

  # Combine datasets along time dimension
  combined_ds = xr.concat(datasets, dim='time')

  # Sort by time to ensure chronological order
  combined_ds = combined_ds.sortby('time')

  print(f"Combined data time range: {combined_ds.time.values[0]} to {combined_ds.time.values[-1]}")
  return combined_ds
def process_era5_data(combined_ds):
    """
    Process ERA5 data for Ethiopia region, limiting to 1981-2023
    """
    # Subset time to 1981-2023
    print("\nSubsetting time period to 1981-2023...")
    ds = combined_ds.sel(time=slice('1981-01-01', '2023-12-31'))

    # Verify the time range after subsetting
    print(f"Time range after subsetting: {ds.time.values[0]} to {ds.time.values[-1]}")

    # Define Ethiopia's bounds
    lat_min, lat_max = 3.5, 14.8
    lon_min, lon_max = 33.0, 47.8

    # Subset data for Ethiopia
    print("\nSubsetting data for Ethiopia region...")
    ds_ethiopia = ds.sel(
        lat=slice(lat_max, lat_min),
        lon=slice(lon_min, lon_max)
    )

    # Remove unnecessary coordinates using drop_vars
    coords_to_drop = ['number', 'expver']
    ds_ethiopia = ds_ethiopia.drop_vars([coord for coord in coords_to_drop if coord in ds_ethiopia])

    print("\nProcessed data dimensions:")
    print(f"Time steps: {len(ds_ethiopia.time)}")
    print(f"Latitude points: {len(ds_ethiopia.lat)}")
    print(f"Longitude points: {len(ds_ethiopia.lon)}")
    print(f"Time range: {ds_ethiopia.time.values[0]} to {ds_ethiopia.time.values[-1]}")

    return ds_ethiopia


def calculate_spi(ds, scale=7):
    """
    Calculate SPI from processed ERA5 data
    """
    print("\nPreparing data for SPI calculation...")

    # Get precipitation data and convert from meters to millimeters
    precip = ds['tp'].values * 1000  # Convert to mm

    # Reshape to 2D (time, space) as required by climate_indices
    original_shape = precip.shape
    time_steps = original_shape[0]
    space_points = original_shape[1] * original_shape[2]
    precip_2d = precip.reshape(time_steps, space_points)

    print(f"Data shape for SPI calculation: {precip_2d.shape}")
    print(f"Number of missing values in precipitation data: {np.isnan(precip).sum()}")

    # Set up SPI parameters with fixed period 1981-2023
    params = {
        'scale': scale,
        'data_start_year': 1981,
        'calibration_year_initial': 1981,
        'calibration_year_final': 2023,
        'periodicity': Periodicity.monthly,
        'distribution': Distribution.gamma
    }

    print("\nCalculating SPI with parameters:")
    print(f"Time period: {params['data_start_year']}-{params['calibration_year_final']}")
    print(f"Scale: {scale} months")

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
    ds['spi'] = (('time', 'lat', 'lon'), spi_3d)
    ds.spi.attrs['long_name'] = f'Standardized Precipitation Index ({scale}-month)'
    ds.spi.attrs['units'] = 'standardized units'
    ds.spi.attrs['calibration_period'] = '1981-2023'

    print(f"SPI calculation complete. Time range in dataset: {ds.time.values[0]} to {ds.time.values[-1]}")

    return ds


def main():
    try:
        # Define input files
        input_files = [
            r'C:\Users\olanrewaju\Downloads\data_0.nc',
            r'C:\Users\olanrewaju\Downloads\data_1.nc'
        ]
        output_file = 'ncfiles/spi7/ethiopia_ERA5_spi_1981_2023.nc'

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Load and combine ERA5 data
        combined_ds = load_and_combine_era5_data(input_files)

        # Process ERA5 data
        ds_processed = process_era5_data(combined_ds)

        # Verify processed time range
        print(f"Processed dataset time range: {ds_processed.time.values[0]} to {ds_processed.time.values[-1]}")

        # Check the length of time steps after processing
        expected_years = 2023 - 1981 + 1
        expected_months = expected_years * 12
        if len(ds_processed.time) < expected_months:
            print(f"Warning: Time dimension has {len(ds_processed.time)} time steps, "
                  f"expected {expected_months} months for {expected_years} years!")

        # Calculate SPI
        print("\nCalculating SPI...")
        ds_with_spi = calculate_spi(ds_processed)

        # Save results
        print(f"\nSaving results to {output_file}")
        encoding = {
            'spi': {'zlib': True, 'complevel': 4, '_FillValue': np.nan},
            'tp': {'zlib': True, 'complevel': 4, '_FillValue': np.nan}
        }
        ds_with_spi.to_netcdf(output_file, encoding=encoding)

        print("\nProcessing completed successfully!")

        # Verify output file
        ds_output = xr.open_dataset(output_file)
        print(f"Saved file time range: {ds_output.time.values[0]} to {ds_output.time.values[-1]}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        raise


if __name__ == "__main__":
    main()