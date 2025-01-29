import os
from subprocess import run


def calculate_climate_index(index, netcdf_precip, netcdf_temp=None, netcdf_pet=None,
                            var_name_precip="precip", var_name_temp="temp", var_name_pet="pet",
                            scales=[7], output_file_base="output",
                            calibration_start_year=1981, calibration_end_year=2023,
                            analysis_start_year=2018, analysis_end_year=2023,
                            periodicity="monthly", multiprocessing="all_but_one"):
    """
    Function to calculate the SPEI-7 index using the `process_climate_indices` entry point.

    Parameters:
    - index (str): The climate index to compute, such as 'spi', 'spei', 'pnp'.
    - netcdf_precip (str): Path to the input NetCDF precipitation file.
    - netcdf_temp (str, optional): Path to the input NetCDF temperature file (required for SPEI).
    - netcdf_pet (str, optional): Path to the input NetCDF PET file (required for SPEI).
    - var_name_precip (str): The variable name for precipitation in the NetCDF file.
    - var_name_temp (str): The variable name for temperature in the NetCDF file (if used).
    - var_name_pet (str): The variable name for PET in the NetCDF file (if used).
    - scales (list): List of time step scales (e.g., [7] for a 7-month scale).
    - output_file_base (str): The base name for the output files.
    - calibration_start_year (int): The start year for the calibration period.
    - calibration_end_year (int): The end year for the calibration period.
    - analysis_start_year (int): The start year for the analysis period (focusing on 2018-2023).
    - analysis_end_year (int): The end year for the analysis period (focusing on 2018-2023).
    - periodicity (str): The time periodicity of the dataset ('monthly' or 'daily').
    - multiprocessing (str): The multiprocessing mode ('all', 'single', or 'all_but_one').

    Returns:
    - None (Runs a subprocess to calculate the climate indices and saves results).
    """

    # Build the command for process_climate_indices
    command = [
                  "process_climate_indices",
                  "--index", index,
                  "--periodicity", periodicity,
                  "--netcdf_precip", netcdf_precip,
                  "--var_name_precip", var_name_precip,
                  "--output_file_base", output_file_base,
                  "--calibration_start_year", str(calibration_start_year),
                  "--calibration_end_year", str(calibration_end_year),
                  "--analysis_start_year", str(analysis_start_year),
                  "--analysis_end_year", str(analysis_end_year),
                  "--scales"] + [str(scale) for scale in scales]  # Add scales dynamically

    if netcdf_temp:
        command.extend(["--netcdf_temp", netcdf_temp, "--var_name_temp", var_name_temp])

    if netcdf_pet:
        command.extend(["--netcdf_pet", netcdf_pet, "--var_name_pet", var_name_pet])

    # Optional parameters
    if multiprocessing:
        command.extend(["--multiprocessing", multiprocessing])

    # Run the command as a subprocess
    print(f"Running command: {' '.join(command)}")
    result = run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("SPEI-7 calculation completed successfully!")
    else:
        print(f"Error: {result.stderr}")


# Paths to your NetCDF files (provided)
netcdf_precip_0 = r'C:\Users\olanrewaju\PycharmProjects\RobustnessCheck\ncfiles\data_0.nc'  # Precipitation NetCDF file 1
netcdf_precip_1 = r'C:\Users\olanrewaju\PycharmProjects\RobustnessCheck\ncfiles\data_1.nc'  # Precipitation NetCDF file 2 (if applicable)
netcdf_temp_path = r'C:\Users\olanrewaju\PycharmProjects\RobustnessCheck\ncfiles\2m temp ERA5.nc'  # Temperature NetCDF file

# Output base directory for results
output_base = "/path/to/output"  # Replace with the actual path for your output

# Example to calculate SPEI-7 index
calculate_climate_index(
    index="spei",  # SPEI index
    netcdf_precip=netcdf_precip_0,  # Precipitation file path
    netcdf_temp=netcdf_temp_path,  # Temperature file path (for SPEI calculation)
    var_name_precip="precipitation",  # Adjust based on your dataset's variable names
    var_name_temp="temperature",  # Adjust based on your dataset's variable names
    scales=[7],  # 7-month scale for SPEI-7
    output_file_base=output_base,
    calibration_start_year=1981,  # Calibration period: 1981-2023
    calibration_end_year=2023,
    analysis_start_year=2018,  # Analysis period: 2018-2023
    analysis_end_year=2023,
    periodicity="monthly",  # Monthly data (adjust if using daily data)
    multiprocessing="all_but_one"  # Parallel processing setting
)
