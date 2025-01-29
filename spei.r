# Open the NetCDF file
library(ncdf4)
nc_file <- nc_open("C:/Users/olanrewaju/PycharmProjects/RobustnessCheck/ncfiles/data_0.nc")

# List all variables and their dimensions
print(nc_file)

# Check the names of the variables in the file
nc_close(nc_file)
