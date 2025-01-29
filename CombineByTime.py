import xarray as xr

# Load the two NetCDF files
data1 = xr.open_dataset(r'C:\Users\olanrewaju\PycharmProjects\RobustnessCheck\ncfiles\Ethiopia_data1.nc', decode_times=False)
data0 = xr.open_dataset(r'C:\Users\olanrewaju\PycharmProjects\RobustnessCheck\ncfiles\Ethiopia_data0.nc', decode_times=False)

# Check the variables to see what dimensions are available
print(data1)
print(data0)

# Ensure both datasets have the same variables and matching dimensions (longitude, latitude)
# Merge the datasets along the time dimension
combined_data = xr.concat([data0, data1], dim='valid_time')

# Verify the concatenation
print(combined_data)

# Optionally, save the combined dataset to a new NetCDF file
combined_data.to_netcdf(r'C:\Users\olanrewaju\PycharmProjects\RobustnessCheck\ncfiles\Ethiopia_combined.nc')
