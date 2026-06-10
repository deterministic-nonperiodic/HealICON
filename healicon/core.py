import glob
import logging
import os
from typing import Optional

import dask
import xarray as xr

from .config import load_variable_mapping, apply_cf_conventions
from .interpolate import HealpixInterpolator

logger = logging.getLogger(__name__)


def process_dataset(
        ds: xr.Dataset,
        dataset_name: str,
        output_file: str,
        nside: Optional[int] = None,
        config_path: Optional[str] = None,
        grid_file: Optional[str] = None,
        use_gpu: bool = False,
        interpolator: Optional[HealpixInterpolator] = None,
        ut_bins: Optional[int] = None
):
    """
    Process a dataset, interpolate to HEALPix, and save to output file.
    """
    logger.info(f"Processing {dataset_name} -> {output_file}")

    # Load mapping
    user_mapping = load_variable_mapping(config_path)

    # Detect SABER data
    is_saber = False
    if ds.attrs.get('Mission') == 'TIMED' and 'SABER' in str(ds.attrs.get('Title', '')):
        logger.info("Detected SABER dataset. Using native parser.")
        from .parsers import parse_saber
        ds = parse_saber(ds, nside=nside, ut_bins=ut_bins)
        is_saber = True

    if grid_file and not is_saber:
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
                logger.warning(f"Variable '{in_var}' not found in {dataset_name}")
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

    # Rechunk: spatial dims fully contiguous, all others automatically chunked (time, height, etc.)
    spatial_dims = []
    for name in ["lon", "longitude", "clon", "lat", "latitude", "clat"]:
        if name in ds.coords or name in ds.data_vars:
            for dim in ds[name].dims:
                if dim not in spatial_dims:
                    spatial_dims.append(dim)

    try:
        from .grid import get_cells_dim
        cell_dim = get_cells_dim(ds)
        if cell_dim not in spatial_dims:
            spatial_dims.append(cell_dim)
    except ValueError:
        pass

    if spatial_dims:
        chunk_dict = {dim: -1 for dim in spatial_dims}
        ds = ds.chunk(chunk_dict).unify_chunks()

    if not is_saber:
        if interpolator is None:
            interpolator = HealpixInterpolator(nside=nside, use_gpu=use_gpu)

        # Perform interpolation
        out_ds = interpolator(ds)
    else:
        out_ds = ds

    # Ensure chunking is reasonable for output writing
    try:
        from .grid import get_cells_dim
        out_cell_dim = get_cells_dim(out_ds)
        out_ds = out_ds.chunk(
            {out_cell_dim: -1})  # Output spatial dimension contiguous for HEALPix maps
    except ValueError:
        pass

    # Save to NetCDF
    logger.info(f"Saving to {output_file}")

    # Use dask delayed for parallel write if needed, but since we chunked, to_netcdf with compute=True handles it.
    # If the dataset is large, we might want to use engine='netcdf4'
    out_ds.to_netcdf(output_file, engine='netcdf4')

    ds.close()
    out_ds.close()
    logger.info(f"Finished {dataset_name}")


def process_file(
        input_file: str,
        output_file: str,
        nside: Optional[int] = None,
        config_path: Optional[str] = None,
        grid_file: Optional[str] = None,
        use_gpu: bool = False,
        interpolator: Optional[HealpixInterpolator] = None,
        ut_bins: Optional[int] = None
):
    """
    Process a single input file, interpolate to HEALPix, and save to output file.
    """
    ds = xr.open_dataset(input_file, chunks='auto')
    process_dataset(ds, input_file, output_file, nside, config_path, grid_file, use_gpu,
                    interpolator, ut_bins)


def run_sequential(
        input_pattern: str,
        output_template: str,
        nside: Optional[int] = None,
        config_path: Optional[str] = None,
        grid_file: Optional[str] = None,
        use_gpu: bool = False,
        ut_bins: Optional[int] = None,
        cat: bool = False
):
    """
    Process multiple files. If cat=True, combine them into a single dataset before processing.
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

    if cat:
        # Use open_mfdataset to combine all input files
        # Check first file to see if it's SABER
        ds_first = xr.open_dataset(input_files[0])
        is_saber = False
        if ds_first.attrs.get('Mission') == 'TIMED' and 'SABER' in str(
                ds_first.attrs.get('Title', '')):
            is_saber = True
        ds_first.close()

        if is_saber:
            max_alt = 0
            for f in input_files:
                with xr.open_dataset(f) as d:
                    max_alt = max(max_alt, d.sizes.get('altitude', 0))

            def preprocess_saber(d):
                import numpy as np
                alt_size = d.sizes.get('altitude', 0)
                if alt_size < max_alt:
                    return d.pad(altitude=(0, max_alt - alt_size), constant_values=np.nan)
                return d

            ds = xr.open_mfdataset(input_files, combine='nested', concat_dim='event', chunks='auto',
                                   preprocess=preprocess_saber)
        else:
            ds = xr.open_mfdataset(input_files, combine='by_coords', chunks='auto')

        # Format output file using the literal template
        output_path = os.path.abspath(output_template)
        for f in input_files:
            if os.path.abspath(f) == output_path:
                logger.error(
                    f"Input file {f} is the same as output file. Aborting to prevent data corruption.")
                return

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        process_dataset(ds, "Combined_MFDataset", output_template, nside, config_path, grid_file,
                        use_gpu, interpolator, ut_bins)
    else:
        for input_file in input_files:
            basename = os.path.basename(input_file)
            # Remove extension for templating
            name_no_ext, _ = os.path.splitext(basename)

            # Format output file
            output_file = output_template.format(basename=basename, name_no_ext=name_no_ext)

            if os.path.abspath(input_file) == os.path.abspath(output_file):
                logger.error(
                    f"Input file {input_file} is the same as output file. Skipping to prevent data corruption.")
                continue

            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)) or '.', exist_ok=True)

            process_file(input_file, output_file, nside, config_path, grid_file, use_gpu,
                         interpolator=interpolator, ut_bins=ut_bins)

    if client:
        client.close()
