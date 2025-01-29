import xarray as xr

# Load the NetCDF file with decode_times=False
file_path = r"C:\Users\olanrewaju\PycharmProjects\RobustnessCheck\ncfiles\2m temp ERA5.nc"
dataset = xr.open_dataset(file_path, decode_times=False)

# Display the dataset's structure to confirm loading
print(dataset)

# Define slicing parameters
lat_min, lat_max = 0, 15  # Latitude range
lon_min, lon_max = 33, 48  # Longitude range

# Slice the dataset
subset = dataset.sel(
    latitude=slice(lat_max, lat_min),  # Latitude is decreasing
    longitude=slice(lon_min, lon_max)
)

# Save the subset to a new NetCDF file
output_path = r"C:\Users\olanrewaju\PycharmProjects\RobustnessCheck\ncfiles\Ethiopia_temp.nc"
subset.to_netcdf(output_path)

print(f"Subset saved to {output_path}")
