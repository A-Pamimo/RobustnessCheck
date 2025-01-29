import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook
import time


def calculate_relative_humidity(t2m, d2m):
    """
    Calculate relative humidity from 2m temperature and dewpoint temperature
    Both inputs should be in Kelvin
    """
    # Constants for Magnus formula
    a = 17.625
    b = 243.04  # °C

    # Convert to Celsius
    t = t2m - 273.15
    td = d2m - 273.15

    # Calculate saturation vapor pressure
    es = 6.112 * np.exp((a * t) / (b + t))
    e = 6.112 * np.exp((a * td) / (b + td))

    # Calculate relative humidity
    rh = (e / es) * 100

    return rh


def calculate_wind_speed(u10, v10):
    """
    Calculate wind speed from U and V components
    """
    return np.sqrt(u10 ** 2 + v10 ** 2)


def process_chunk(chunk_data):
    """
    Process a chunk of data with all calculations
    """
    result = {}

    # Temperature conversion
    if 't2m' in chunk_data:
        result['t2m'] = chunk_data['t2m'] - 273.15

    # Relative humidity
    if 't2m' in chunk_data and 'd2m' in chunk_data:
        result['rh'] = calculate_relative_humidity(
            chunk_data['t2m'] + 273.15,
            chunk_data['d2m']
        )

    # Wind speed
    if 'u10' in chunk_data and 'v10' in chunk_data:
        result['wind_speed'] = calculate_wind_speed(
            chunk_data['u10'],
            chunk_data['v10']
        )

    # Radiation conversion
    for var in ['ssrd', 'strd']:
        if var in chunk_data:
            result[var] = chunk_data[var] / 1e6

    return result


def process_era5_data(data_files):
    """
    Process ERA5 data files for SPEI calculation with multithreading and progress tracking
    """
    print("\nStarting ERA5 data processing pipeline...")

    # Load datasets with progress bar
    ds_list = []
    with tqdm(total=len(data_files), desc="Loading files") as pbar:
        for file in data_files:
            pbar.set_description(f"Loading {Path(file).name}")
            ds = xr.open_dataset(file)
            ds_list.append(ds)
            pbar.update(1)

    # Merge datasets
    print("\nMerging datasets...")
    ds = xr.merge(ds_list)
    print(f"Merged dataset dimensions: {ds.dims}")

    # Subset to Ethiopia
    print("\nSubsetting to Ethiopia region...")
    ds_ethiopia = ds.sel(
        latitude=slice(14.8, 3.5),
        longitude=slice(33.0, 47.8)
    )
    print(f"Ethiopia subset dimensions: {ds_ethiopia.dims}")

    # Get the number of CPU cores
    num_cores = multiprocessing.cpu_count()
    chunk_size = 100

    # Split data into chunks
    lat_chunks = np.array_split(ds_ethiopia.latitude, max(1, len(ds_ethiopia.latitude) // chunk_size))

    print(f"\nProcessing data using {num_cores} cores...")
    print(f"Total chunks to process: {len(lat_chunks)}")

    # Process chunks in parallel with progress bar
    processed_chunks = []
    start_time = time.time()
    with tqdm(total=len(lat_chunks), desc="Processing chunks") as pbar:
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            futures = []
            for i, lat_chunk in enumerate(lat_chunks):
                chunk_data = ds_ethiopia.sel(latitude=lat_chunk)
                futures.append(executor.submit(process_chunk, chunk_data))

            for future in as_completed(futures):
                processed_chunks.append(future.result())
                pbar.update(1)

                # Update progress details
                elapsed = time.time() - start_time
                chunks_done = len(processed_chunks)
                chunks_left = len(lat_chunks) - chunks_done
                avg_time_per_chunk = elapsed / chunks_done
                eta = avg_time_per_chunk * chunks_left

                pbar.set_postfix({
                    'Done': f"{chunks_done}/{len(lat_chunks)}",
                    'ETA': f"{eta:.1f}s"
                })

    # Combine processed chunks
    print("\nCombining processed chunks...")
    ds_processed = xr.concat(processed_chunks, dim='latitude')

    # Update attributes with progress tracking
    print("\nUpdating variable attributes...")
    with tqdm(total=7, desc="Updating attributes") as pbar:
        if 't2m' in ds_processed:
            ds_processed['t2m'].attrs['units'] = 'degC'
            pbar.update(1)

        if 'rh' in ds_processed:
            ds_processed['rh'].attrs.update({
                'units': '%',
                'long_name': 'Relative humidity'
            })
            pbar.update(1)

        if 'wind_speed' in ds_processed:
            ds_processed['wind_speed'].attrs.update({
                'units': 'm s-1',
                'long_name': 'Wind speed at 10m'
            })
            pbar.update(1)

        for var in ['ssrd', 'strd']:
            if var in ds_processed:
                ds_processed[var].attrs['units'] = 'MJ m-2 day-1'
                pbar.update(1)

        # Remove unnecessary coordinates
        if 'number' in ds_processed:
            ds_processed = ds_processed.drop_vars('number')
            pbar.update(1)
        if 'expver' in ds_processed:
            ds_processed = ds_processed.drop_vars('expver')
            pbar.update(1)

    # Print processing summary
    print("\nProcessing Summary:")
    print(f"Total processing time: {time.time() - start_time:.1f} seconds")
    print(f"Processed {len(lat_chunks)} chunks using {num_cores} cores")
    print(f"Final dataset dimensions: {ds_processed.dims}")

    return ds_processed


def save_processed_data(ds, output_file):
    """
    Save processed dataset with proper encoding and progress tracking
    """
    print(f"\nPreparing to save data to {output_file}")

    # Setup encoding
    encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}

    # Calculate approximate file size
    total_size = sum(ds[var].nbytes for var in ds.data_vars) / (1024 * 1024)  # MB
    print(f"Estimated uncompressed size: {total_size:.1f} MB")

    # Save with progress tracking
    print("Saving data (this may take a while)...")
    start_time = time.time()

    ds.to_netcdf(
        output_file,
        encoding=encoding,
        compute=True,
        engine='netcdf4'
    )

    elapsed = time.time() - start_time
    print(f"Save completed in {elapsed:.1f} seconds")


def main():
    # Input files
    data_files = [
        r'C:\Users\olanrewaju\Downloads\PET\data_0.nc',
        r'C:\Users\olanrewaju\Downloads\PET\data_1.nc',
        r'C:\Users\olanrewaju\Downloads\PET\data_2.nc'
    ]

    output_file = 'ethiopia_spei_input.nc'

    try:
        # Process data
        print("\n=== Starting ERA5 Data Processing ===")
        start_time = time.time()

        ds_processed = process_era5_data(data_files)

        # Print summary
        print("\nProcessed Data Summary:")
        print(f"Dimensions: {ds_processed.dims}")
        print("Variables:", list(ds_processed.data_vars))
        print(f"Time range: {ds_processed.valid_time.values[0]} to {ds_processed.valid_time.values[-1]}")

        # Save processed data
        save_processed_data(ds_processed, output_file)

        total_time = time.time() - start_time
        print(f"\nTotal processing time: {total_time:.1f} seconds")
        print("Processing completed successfully!")

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()