import glob
import logging
import math
import os
from typing import Optional

import dask
import xarray as xr

from .config import load_variable_mapping, apply_cf_conventions
from .interpolate import HealpixInterpolator

logger = logging.getLogger(__name__)


def process_file(
        input_file: str,
        output_file: str,
        nside: Optional[int] = None,
        config_path: Optional[str] = None,
        grid_file: Optional[str] = None,
        use_gpu: bool = False,
        interpolator: Optional[HealpixInterpolator] = None
):
    """
    Process a single input file, interpolate to HEALPix, and save to output file.
    """
    logger.info(f"Processing {input_file} -> {output_file}")

    # Load mapping
    user_mapping = load_variable_mapping(config_path)

    # Open dataset with chunking for Dask
    ds = xr.open_dataset(input_file, chunks='auto')

    if grid_file:
        logger.info(f"Loading external grid file: {grid_file}")
        grid_ds = xr.open_dataset(grid_file)
        grid_coords = {}
        for name in ["lon", "longitude", "clon", "lat", "latitude", "clat"]:
            if name in grid_ds.variables:
                grid_coords[name] = grid_ds[name]
        ds = ds.assign_coords(grid_coords)

    # Select variables based on mapping or CF conventions
    vars_to_keep = []

    if user_mapping:
        for out_var, in_var in user_mapping.items():
            if in_var in ds:
                # Rename the variable if needed
                if out_var != in_var:
                    ds = ds.rename({in_var: out_var})
                vars_to_keep.append(out_var)
            else:
                logger.warning(f"Variable '{in_var}' not found in {input_file}")
    else:
        # Fallback to CF conventions
        cf_map = apply_cf_conventions(ds)
        if cf_map:
            logger.info(f"Found CF variables: {list(cf_map.keys())}")
            vars_to_keep = list(cf_map.values())
        else:
            logger.warning(
                "No variable mapping provided and no CF conventions found. Processing all data variables.")
            vars_to_keep = list(ds.data_vars.keys())

    # Subset dataset
    if vars_to_keep:
        # Also keep coordinates needed for interpolation
        coords_to_keep = []
        for name in ["lon", "longitude", "clon", "lat", "latitude", "clat"]:
            if name in ds.coords or name in ds.data_vars:
                coords_to_keep.append(name)

        all_vars = vars_to_keep + coords_to_keep
        # Drop variables not in the list
        drop_vars = [v for v in ds.variables if v not in all_vars and v not in ds.dims]
        ds = ds.drop_vars(drop_vars, errors='ignore')

    # Rechunk: spatial dims fully contiguous, all others chunked by 1 (time, height, etc.)
    spatial_dims = []
    for name in ["lon", "longitude", "clon", "lat", "latitude", "clat"]:
        if name in ds.coords or name in ds.data_vars:
            for dim in ds[name].dims:
                if dim not in spatial_dims:
                    spatial_dims.append(dim)

    if spatial_dims:
        chunk_dict = {dim: -1 for dim in spatial_dims}
        for dim in ds.dims:
            if dim not in spatial_dims:
                chunk_dict[dim] = 1
        ds = ds.chunk(chunk_dict)

    if interpolator is None:
        interpolator = HealpixInterpolator(nside=nside, use_gpu=use_gpu)

    # Perform interpolation
    out_ds = interpolator(ds)

    # Ensure chunking is reasonable for output writing
    try:
        from .grid import get_cells_dim
        out_cell_dim = get_cells_dim(out_ds)
        out_ds = out_ds.chunk({out_cell_dim: -1})  # Output spatial dimension contiguous for HEALPix maps
    except ValueError:
        pass

    # Save to NetCDF
    logger.info(f"Saving to {output_file}")

    # Use dask delayed for parallel write if needed, but since we chunked, to_netcdf with compute=True handles it.
    # If the dataset is large, we might want to use engine='netcdf4'
    out_ds.to_netcdf(output_file, engine='netcdf4')

    ds.close()
    out_ds.close()
    logger.info(f"Finished {input_file}")


def run_sequential(
        input_pattern: str,
        output_template: str,
        nside: Optional[int] = None,
        config_path: Optional[str] = None,
        grid_file: Optional[str] = None,
        use_gpu: bool = False
):
    """
    Process multiple files sequentially.
    """
    input_files = sorted(glob.glob(input_pattern))

    if not input_files:
        logger.error(f"No files found matching {input_pattern}")
        return

    logger.info(f"Found {len(input_files)} files to process.")

    from dask.distributed import Client
    try:
        # Check if a client exists
        client = dask.distributed.client.default_client()
        logger.info(f"Using existing Dask client: {client.dashboard_link}")
    except ValueError:
        # Use Dask's default threaded scheduler for single-node execution.
        # This avoids massive memory duplication across processes and handles 
        # sequential to_netcdf writes much more robustly without OOM.
        logger.info("Using default Dask threaded scheduler.")
        client = None

    interpolator = HealpixInterpolator(nside=nside, use_gpu=use_gpu)

    for input_file in input_files:
        basename = os.path.basename(input_file)
        # Remove extension for templating
        name_no_ext, _ = os.path.splitext(basename)

        # Format output file
        output_file = output_template.format(basename=basename, name_no_ext=name_no_ext)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        process_file(input_file, output_file, nside, config_path, grid_file, use_gpu, interpolator=interpolator)

    if client:
        client.close()
