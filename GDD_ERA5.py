import xarray as xr
import numpy as np


def calculate_gdd(ds, base_temp=10.0):
    """
    Calculate Growing Degree Days (GDD) from temperature data

    Parameters:
    ds : xarray Dataset containing temperature data
    base_temp : float, base temperature in Celsius (default 10°C)

    Returns:
    xarray DataArray with GDD values
    """
    # Convert temperature from Kelvin to Celsius
    temp_celsius = ds.t2m - 273.15

    # Calculate GDD: max(0, T - base_temp)
    # Where T is the daily mean temperature
    gdd = temp_celsius - base_temp
    gdd = gdd.where(gdd > 0, 0)  # Replace negative values with 0

    return gdd


def main():
    # Read the ERA5 temperature data
    ds = xr.open_dataset('ncfiles/2m temp ERA5.nc')

    # Define Ethiopia's approximate bounding box
    ethiopia_bounds = {
        'latitude': slice(15, 3),  # North to South
        'longitude': slice(33, 48)  # West to East
    }

    # Subset data for Ethiopia
    ds_ethiopia = ds.sel(**ethiopia_bounds)

    # Calculate GDD
    gdd = calculate_gdd(ds_ethiopia)

    # Create a new dataset with GDD values
    ds_out = xr.Dataset(
        {
            'gdd': (('valid_time', 'latitude', 'longitude'), gdd.values),
        },
        coords={
            'valid_time': ds_ethiopia.valid_time,
            'latitude': ds_ethiopia.latitude,
            'longitude': ds_ethiopia.longitude
        }
    )

    # Add metadata
    ds_out.gdd.attrs = {
        'units': 'degree_days',
        'long_name': 'Growing Degree Days',
        'description': 'Calculated with base temperature of 10°C',
    }

    # Save to NetCDF file
    ds_out.to_netcdf('ethiopia_gdd.nc')


if __name__ == '__main__':
    main()