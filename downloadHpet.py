import cdsapi
import os


def download_era5_variables():
    """
    Download ERA5 variables needed for SPEI calculation
    """
    c = cdsapi.Client()

    # Define variables to download
    variables = [
        '2m_temperature',
        '2m_dewpoint_temperature',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        'surface_solar_radiation_downwards',
        'surface_thermal_radiation_downwards',
        'surface_pressure',
        'total_cloud_cover',
    ]

    # Area for Ethiopia [North, West, South, East]
    area = [14.8, 33.0, 3.5, 47.8]

    for var in variables:
        output_file = f'era5_{var}_1981_2023.nc'

        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping download.")
            continue

        print(f"\nDownloading {var}...")

        c.retrieve(
            'reanalysis-era5-single-levels-monthly-means',
            {
                'product_type': 'monthly_averaged_reanalysis',
                'variable': var,
                'year': [str(year) for year in range(1981, 2024)],
                'month': [str(month).zfill(2) for month in range(1, 13)],
                'time': '00:00',
                'area': area,
                'format': 'netcdf',
            },
            output_file
        )
        print(f"Downloaded {var} successfully!")


if __name__ == "__main__":
    print("Starting ERA5 downloads for SPEI calculation...")
    try:
        download_era5_variables()
        print("\nAll downloads completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")