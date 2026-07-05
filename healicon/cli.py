import functools
import glob
import logging
import os
import resource
import time

import click

from .cf_coords import _cf_guess
from .core import run_sequential
from .grid import get_spatial_dims
from .io_utils import write_dataset


def profile_command(func):
    """Decorator to add CDO-style profiling output to CLI commands."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        result = func(*args, **kwargs)

        end_time = time.time()
        elapsed = end_time - start_time

        # Get peak memory
        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mem_used_mb = ru_maxrss / 1024.0

        # Extract metadata from ifile
        ifile_path = kwargs.get('ifile')

        n_vars, n_time, n_levels = 0, 0, 0
        if ifile_path and isinstance(ifile_path, str):
            files = glob.glob(ifile_path)
            if files:
                import xarray as xr
                try:
                    with xr.open_dataset(files[0]) as ds:
                        vars_list = [v for v in ds.data_vars if ds[v].ndim > 0]
                        n_vars = len(vars_list)

                        time_dims = [d for d in ds.dims if d.lower() in ('time', 'lst')]
                        n_time = ds.sizes[time_dims[0]] if time_dims else 1
                        n_time *= len(files)

                        level_dims = [d for d in ds.dims if
                                      d.lower() in ('level', 'height', 'z_mc', 'lev')]
                        n_levels = ds.sizes[level_dims[0]] if level_dims else 1
                except Exception:
                    pass

        var_str = f"1 variable" if n_vars == 1 else f"{n_vars} variables"
        time_str = f"1 timestep" if n_time == 1 else f"{n_time} timesteps"
        lev_str = f"1 level" if n_levels == 1 else f"{n_levels} levels"

        click.echo(
            f"Processed {var_str} over {time_str} {lev_str} [{elapsed:.2f}s {mem_used_mb:.0f}MB]")

        return result

    return wrapper


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress overly verbose healpy INFO logs
logging.getLogger('healpy').setLevel(logging.WARNING)


def _check_io_safety(ifile, ofile):
    """Raise if input and output resolve to the same file."""
    if os.path.abspath(ifile) == os.path.abspath(ofile):
        raise click.UsageError(
            "Input and output files cannot be the same. This would corrupt the input file.")


def _guess_variable(ds, target_type: str) -> str:
    name = _cf_guess(ds, target_type)
    if name is not None:
        logger.info(f"Auto-detected {target_type} variable: '{name}'")
        return name

    if len(ds.data_vars) == 1:
        name = list(ds.data_vars.keys())[0]
        logger.info(f"Auto-detected {target_type} variable (only var in dataset): '{name}'")
        return name

    raise ValueError(
        f"Could not automatically detect {target_type} variable. Please specify it explicitly.")


@click.group()
def cli():
    """HealICON: Interpolate atmospheric model outputs to HEALPix grid."""
    try:
        from dask.diagnostics import ProgressBar
        ProgressBar().register()
    except ImportError:
        pass


def _load_and_ensure_healpix(ifile, target_nside=None):
    """
    Load a dataset and ensure it is on a HEALPix grid.
    
    Args:
        ifile: Input file path.
        target_nside: Target NSIDE for the HEALPix grid.
    """
    import xarray as xr
    from .interpolate import interpolate_to_healpix

    logger.info(f"Opening file: {ifile}")
    ds = xr.open_dataset(ifile, chunks='auto')

    # Optimize Dask chunking: Spatial dimensions must be fully contiguous for spectral analysis
    spatial_dims = get_spatial_dims(ds)

    if spatial_dims:
        ds = ds.chunk({dim: -1 for dim in spatial_dims}).unify_chunks()

    from .grid import is_healpix as _is_healpix

    if not _is_healpix(ds):
        # Detect SABER data
        if ds.attrs.get('Mission') == 'TIMED' and 'SABER' in str(ds.attrs.get('Title', '')):
            logger.info("Detected SABER dataset. Using native parser.")
            from .parsers import parse_saber
            ds = parse_saber(ds, nside=target_nside)
        else:
            logger.info("Input dataset is not a HEALPix grid. Auto-interpolating first...")
            ds = interpolate_to_healpix(ds, nside=target_nside)

    return ds


@cli.command()
@click.argument('ifile', type=str)
@click.argument('ofile')
@click.option('-n', '--nside', type=int, required=False, default=None,
              help='HEALPix Nside resolution parameter (e.g., 32, 64, 128). If omitted, defaults to closest resolution.')
@click.option('--ut-bins', type=int, default=None,
              help='Number of Universal Time bins (e.g. 24). Only applicable for native parsers (like SABER) that support UT binning.')
@click.option('-c', '--config', 'config_path', type=click.Path(exists=True), default=None,
              help='Path to YAML configuration file for variable mapping.')
@click.option('-g', '--grid', 'grid_file', type=click.Path(exists=True), default=None,
              help='Optional path to external grid file containing coordinates (e.g., clat, clon).')
@click.option('--gpu', is_flag=True, default=False,
              help='Enable GPU acceleration for KDTree interpolation if available.')
@click.option('--cat', is_flag=True, default=False,
              help='Combine files matching the input pattern into a single dataset before processing (mimics CDO cat).')
@profile_command
def convert(ifile, ofile, nside, ut_bins, config_path, grid_file, gpu, cat):
    """
    Convert model output to HEALPix grid.

    ifile: Path or wildcard pattern to input model output file(s) (NetCDF).
    ofile: Path to output HEALPix file (NetCDF).
    nside: HEALPix Nside resolution parameter (e.g., 32, 64, 128).
           If omitted, defaults to closest resolution.
    config_path: Optional path to YAML configuration file for variable mapping.
    grid_file: Optional path to external grid file containing coordinates.
    gpu: Enable GPU acceleration for KDTree interpolation if available.
    """
    _check_io_safety(ifile, ofile)

    run_sequential(
        input_pattern=ifile,
        output_template=ofile,
        nside=nside,
        config_path=config_path,
        grid_file=grid_file,
        use_gpu=gpu,
        ut_bins=ut_bins,
        cat=cat
    )


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('-l', '--lat', type=float, required=True,
              help='Target latitude in degrees [-90, 90].')
@click.option('--num-lons', type=int, default=None,
              help='Number of longitude points to extract (default: number of HEALPix grid points).')
@profile_command
def extract_lat(ifile, ofile, lat, num_lons):
    """
    Extract data along all longitudes for a specific latitude from a HEALPix dataset.

    ifile: Path or wildcard pattern to input model output file(s) (NetCDF).
    ofile: Path to output HEALPix file (NetCDF).
    lat: Target latitude in degrees [-90, 90].
    num_lons: Number of longitude points to extract (default: number of HEALPix grid points).
    """
    _check_io_safety(ifile, ofile)

    from .extract import extract_along_latitude

    ds = _load_and_ensure_healpix(ifile)

    out_ds = extract_along_latitude(ds, lat=lat, num_lons=num_lons)

    logger.info(f"Computing and saving extracted data to {ofile}")
    write_dataset(out_ds, ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('-l', '--lon', type=float, required=True,
              help='Target longitude in degrees [-180, 180] or [0, 360].')
@click.option('--num-lats', type=int, default=None,
              help='Number of latitude points to extract.')
@profile_command
def extract_lon(ifile, ofile, lon, num_lats):
    """
    Extract data along all latitudes for a specific longitude from a HEALPix dataset.

    Args:
        ifile: Path or wildcard pattern to input model output file(s) (NetCDF).
        ofile: Path to output HEALPix file (NetCDF).
        lon: Target longitude in degrees [-180, 180] or [0, 360].
        num_lats: Number of latitude points to extract (default: number of HEALPix grid points).
    """
    _check_io_safety(ifile, ofile)

    from .extract import extract_along_longitude

    ds = _load_and_ensure_healpix(ifile)
    out_ds = extract_along_longitude(ds, lon=lon, num_lats=num_lats)
    logger.info(f"Computing and saving extracted data to {ofile}")
    write_dataset(out_ds, ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('--lat', type=float, required=True, help='Target latitude.')
@click.option('--lon', type=float, required=True, help='Target longitude.')
@profile_command
def extract_point(ifile, ofile, lat, lon):
    """
    Extract full time/height profile for a specific lat/lon point from a HEALPix dataset.

    Args:
        ifile: Path or wildcard pattern to input model output file(s) (NetCDF).
        ofile: Path to output HEALPix file (NetCDF).
        lat: Target latitude in degrees [-90, 90].
        lon: Target longitude in degrees [-180, 180] or [0, 360].
    """
    _check_io_safety(ifile, ofile)

    from .extract import extract_point as ep

    ds = _load_and_ensure_healpix(ifile)
    out_ds = ep(ds, lat=lat, lon=lon)
    logger.info(f"Computing and saving extracted data to {ofile}")
    write_dataset(out_ds, ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('--spatial-dim', type=str, default='cells',
              help='Name of the spatial dimension (default: cells).')
@click.option('--time-dim', type=str, default=None,
              help='Optional name of the time/LST dimension to interpolate across temporally.')
@profile_command
def fill(ifile, ofile, spatial_dim, time_dim):
    """
    Fill missing values (NaNs) natively using HEALPix KDTree spatial nearest-neighbor and temporal 1D linear interpolation.

    Args:
        ifile: Path to input data file.
        ofile: Path to output data file.
        spatial_dim: Name of the spatial dimension (default: cells).
        time_dim: Optional name of the time/LST dimension to interpolate across temporally.
    """
    _check_io_safety(ifile, ofile)

    from .curation import fill_healpix_gaps
    import xarray as xr

    # Load dataset
    ds = xr.open_dataset(ifile, chunks='auto')

    out_ds = fill_healpix_gaps(ds, spatial_dim=spatial_dim, time_dim=time_dim)

    logger.info(f"Saving filled dataset to {ofile}")
    write_dataset(out_ds, ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@profile_command
def zonal_mean(ifile, ofile):
    """
    Compute the zonal mean (longitude average) across HEALPix rings.

    Args:
        ifile: Path to input data file.
        ofile: Path to output data file.
    """
    _check_io_safety(ifile, ofile)

    from .extract import zonal_mean as zm

    ds = _load_and_ensure_healpix(ifile)
    out_ds = zm(ds)
    logger.info(f"Computing and saving zonal mean to {ofile}")
    write_dataset(out_ds, ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('-v', '--var', 'var_name', default=None, multiple=True,
              help='Variable(s) to compute spectrum for. Can be specified multiple times.')
@click.option('--lmax', type=int, default=None,
              help='Maximum spherical harmonic degree l.')
@click.option('--type', 'spectrum_type', type=click.Choice(['power', 'cross', 'kinetic']),
              default='power',
              help='Type of spectrum to compute: power (default), cross, or kinetic.')
@profile_command
def spectrum(ifile, ofile, var_name, lmax, spectrum_type):
    """
    Compute the angular spectrum (Cl) of a variable or pair of variables.

    Args:
        ifile: Path to input data file.
        ofile: Path to output data file.
        var_name: Variable(s) to compute spectrum for. Can be specified multiple times.
        lmax: Maximum spherical harmonic degree l.
        spectrum_type: Type of spectrum to compute: power (default), cross, or kinetic.
    """
    _check_io_safety(ifile, ofile)

    from .analysis import compute_spectrum, degree_to_wavelength

    ds = _load_and_ensure_healpix(ifile)

    # Pass empty list if no variables provided (triggers auto-detection)
    var_list = list(var_name) if var_name else None

    out_ds = compute_spectrum(ds, var_name=var_list, lmax=lmax, spectrum_type=spectrum_type)
    logger.info(f"Computing and saving {spectrum_type} spectrum to {ofile}")
    out_ds = out_ds.compute()
    # Log effective resolution after computing
    actual_lmax = int(out_ds['l'].max())
    nyquist_km = degree_to_wavelength(actual_lmax)
    logger.info(f"Spectrum resolved up to lmax={actual_lmax} (~{nyquist_km:.0f} km Nyquist scale).")
    write_dataset(out_ds, ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('--fwhm', type=float, default=None,
              help='Full-width half-max in degrees for Gaussian smoothing.')
@click.option('--lmax', type=int, default=None,
              help='Hard low-pass spectral cutoff at spherical harmonic degree l.')
@click.option('--wavelength', 'wavelength_km', type=float, default=None,
              help='Hard low-pass spectral cutoff expressed as a physical wavelength in km.')
@profile_command
def filter(ifile, ofile, fwhm, lmax, wavelength_km):
    """
    Filter spatial maps using spherical harmonic transforms.

    Args:
        ifile: Path to input data file.
        ofile: Path to output data file.
        fwhm: Full-width half-max in degrees for Gaussian smoothing.
        lmax: Hard low-pass spectral cutoff at spherical harmonic degree l.
        wavelength_km: Hard low-pass spectral cutoff expressed as a physical wavelength in km.
    """
    _check_io_safety(ifile, ofile)

    from .analysis import filter_spatial

    ds = _load_and_ensure_healpix(ifile)
    out_ds = filter_spatial(ds, fwhm_deg=fwhm, lmax=lmax, wavelength_km=wavelength_km)
    logger.info(f"Computing and saving filtered data to {ofile}")
    write_dataset(out_ds, ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('-n', '--nside', type=int, default=None,
              help='Target nside for the resolution change.')
@click.option('-z', '--zoom', type=int, default=None,
              help='Target zoom level (refinement), where nside = 2**zoom.')
@profile_command
def regrade(ifile, ofile, nside, zoom):
    """
    Upgrade or downgrade the HEALPix resolution.

    Args:
        ifile: Path to input data file.
        ofile: Path to output data file.
        nside: Target nside for the resolution change.
        zoom: Target zoom level (refinement), where nside = 2**zoom.
    """
    _check_io_safety(ifile, ofile)

    from .analysis import regrade_resolution

    if nside is None and zoom is None:
        raise click.UsageError("You must provide either --nside or --zoom.")
    if zoom is not None:
        nside = 2 ** zoom

    ds = _load_and_ensure_healpix(ifile, target_nside=nside)

    # Check if we still need to regrade (if auto-interp already hit nside, skip ud_grade to save time)
    import healpy as hp
    try:
        from .grid import get_cells_dim
        cell_dim = get_cells_dim(ds)
        if ds.sizes[cell_dim] == hp.nside2npix(nside):
            out_ds = ds
        else:
            out_ds = regrade_resolution(ds, new_nside=nside)
    except ValueError:
        out_ds = regrade_resolution(ds, new_nside=nside)
    logger.info(f"Computing and saving regraded data to {ofile}")
    write_dataset(out_ds, ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('--u', 'u_var', default=None, help='Name of eastward wind variable.')
@click.option('--v', 'v_var', default=None, help='Name of northward wind variable.')
@click.option('--lmax', type=int, default=None, help='Max spherical harmonic degree.')
@profile_command
def uv2dv(ifile, ofile, u_var, v_var, lmax):
    """
    Compute horizontal divergence and vorticity from U and V wind components.

    Args:
        ifile: Path to input data file.
        ofile: Path to output data file.
        u_var: Name of eastward wind variable (default: detected from file).
        v_var: Name of northward wind variable (default: detected from file).
        lmax: Maximum spherical harmonic degree l (default: all available).
    """
    _check_io_safety(ifile, ofile)

    from .analysis import compute_vorticity_divergence

    ds = _load_and_ensure_healpix(ifile)
    if u_var is None:
        u_var = _guess_variable(ds, "u")
    if v_var is None:
        v_var = _guess_variable(ds, "v")
    out_ds = compute_vorticity_divergence(ds, u_var=u_var, v_var=v_var, lmax=lmax)
    logger.info(f"Computing and saving vorticity/divergence to {ofile}")
    write_dataset(out_ds, ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('--div', 'div_var', default=None, help='Name of divergence variable.')
@click.option('--vor', 'vor_var', default=None, help='Name of vorticity variable.')
@click.option('--lmax', type=int, default=None, help='Max spherical harmonic degree.')
@profile_command
def dv2uv(ifile, ofile, div_var, vor_var, lmax):
    """
    Compute U and V wind components from horizontal divergence and vorticity.

    Args:
        ifile: Path to input data file.
        ofile: Path to output data file.
        div_var: Name of divergence variable (default: detected from file).
        vor_var: Name of vorticity variable (default: detected from file).
        lmax: Maximum spherical harmonic degree l (default: all available).
    """
    _check_io_safety(ifile, ofile)

    from .analysis import compute_uv_from_vorticity_divergence

    ds = _load_and_ensure_healpix(ifile)
    if div_var is None:
        div_var = _guess_variable(ds, "divergence")
    if vor_var is None:
        vor_var = _guess_variable(ds, "vorticity")
    out_ds = compute_uv_from_vorticity_divergence(ds, div_var=div_var, vor_var=vor_var, lmax=lmax)
    logger.info(f"Computing and saving U/V to {ofile}")
    write_dataset(out_ds, ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('--u', 'u_var', default=None, help='Name of eastward wind variable.')
@click.option('--v', 'v_var', default=None, help='Name of northward wind variable.')
@click.option('--lmax', type=int, default=None, help='Max spherical harmonic degree.')
@click.option('--psi/--no-psi', default=True, show_default=True,
              help='Include streamfunction ψ [m² s⁻¹] in output.')
@click.option('--chi/--no-chi', default=True, show_default=True,
              help='Include velocity potential χ [m² s⁻¹] in output.')
@profile_command
def helmholtz(ifile, ofile, u_var, v_var, lmax, psi, chi):
    """
    Helmholtz decomposition: split wind into rotational and divergent components.

    Outputs u_rot, v_rot (rotational wind) and u_div, v_div (divergent wind).
    Optionally also computes the streamfunction (--psi) and velocity potential
    (--chi), both in units of m² s⁻¹.

    Args:
        ifile: Path to input data file.
        ofile: Path to output data file.
        u_var: Name of eastward wind variable (default: detected from file).
        v_var: Name of northward wind variable (default: detected from file).
        lmax: Maximum spherical harmonic degree l (default: all available).
        psi: Include streamfunction ψ [m² s⁻¹] in output.
        chi: Include velocity potential χ [m² s⁻¹] in output.
    """
    _check_io_safety(ifile, ofile)

    from .analysis import compute_helmholtz

    ds = _load_and_ensure_healpix(ifile)
    if u_var is None:
        u_var = _guess_variable(ds, "u")
    if v_var is None:
        v_var = _guess_variable(ds, "v")
    out_ds = compute_helmholtz(ds, u_var=u_var, v_var=v_var, lmax=lmax,
                               include_psi=psi, include_chi=chi)
    logger.info(f"Computing and saving Helmholtz decomposition to {ofile}")
    write_dataset(out_ds, ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('-v', '--var', 'var_name', default=None,
              help='Variable to compute tidal analysis for.')
@click.option('-p', '--periods', 'periods_str', default=None,
              help='Comma-separated tidal periods in hours. E.g., 12.0,24.0 for semidiurnal and diurnal.')
@click.option('-m', '--m-filters', 'm_str', default=None,
              help='Optional comma-separated zonal wavenumbers. E.g., 1,2,3. '
                   'Positive values denote westward propagation, negative values eastward '
                   '(matching Yamazaki 2023 convention, consistent across all methods).')
@click.option('--modes', 'modes_str', default=None,
              help='Comma-separated tidal modes. E.g., DW1, SW2, DE3, SE2. '
                   'If provided, automatically configures periods and m-filters.')
@click.option('--lmax', type=int, default=None, help='Max spherical harmonic degree.')
@click.option('--time-dim', default='time', show_default=True,
              help='Dimension name for time-like axis (e.g., "time" or "lst").')
@click.option('--method', default='ls', type=click.Choice(['ls', 'fourier', 'sh']),
              show_default=True, help='Tidal analysis method to use.')
@click.option('--temporal-mean/--no-temporal-mean', default=True, show_default=True,
              help='Average wavelet/fourier amplitudes over time (comparable to LS).')
@click.option('--dj', type=float, default=0.1, show_default=True,
              help='Spacing between discrete wavelet scales (for fourier and wavelet methods).')
@click.option('--no-sym-asy', 'decompose_sym_asy', is_flag=True, default=True,
              flag_value=False,
              help='Skip symmetric/antisymmetric decomposition; output the total tidal field directly.')
@profile_command
def tides(ifile, ofile, var_name, periods_str, m_str, modes_str, lmax, time_dim, method,
          temporal_mean, dj, decompose_sym_asy):
    """
    Perform a full tidal analysis (temporal fit and spatial symmetry decomposition).

    Extracts the amplitude and phase of given periods (in hours), optionally 
    filters to specific zonal wavenumbers (-m), and decomposes the resulting 
    spatial patterns into symmetric and antisymmetric components relative to the equator.

    Note on Sign Convention:
      - All methods (ls, fourier, wavelet) share a consistent convention:
        positive m = westward, negative m = eastward (complying with Yamazaki 2023).
      - Using the --modes option (e.g. --modes DW1) automatically handles this.

    Args:
        ifile: Path to input data file.
        ofile: Path to output data file.
        var_name: Variable to compute tidal analysis for.
        periods_str: Comma-separated tidal periods in hours (e.g., "12.0,24.0").
        m_str: Optional comma-separated zonal wavenumbers (e.g., "1,2,3").
        modes_str: Comma-separated tidal modes (e.g., "DW1, SW2, DE3, SE2").
        lmax: Maximum spherical harmonic degree l.
        time_dim: Dimension name for time-like axis (e.g., "time" or "lst").
        method: Tidal analysis method (ls, fourier, or wavelet).
        temporal_mean: Whether to average wavelet/fourier amplitude over time.
        dj: Discrete wavelet scale spacing.
    """
    _check_io_safety(ifile, ofile)

    if modes_str:
        if periods_str or m_str:
            raise click.UsageError("Cannot specify --periods or -m when --modes is used.")

        periods_hours, m_filters = [], []
        period_map = {'D': 24.0, 'S': 12.0, 'T': 8.0, 'Q': 6.0}

        for mode in [m.strip().upper() for m in modes_str.split(',')]:
            if not mode: continue

            p_char = mode[0]
            if p_char not in period_map:
                raise click.UsageError(
                    f"Unknown period identifier '{p_char}' in mode '{mode}'. Use D (24h), S (12h), T (8h), Q (6h).")
            periods_hours.append(period_map[p_char])

            if len(mode) < 2:
                raise click.UsageError(f"Invalid mode format '{mode}'.")

            if mode[1] == 'W':
                m_filters.append(int(mode[2:]))
            elif mode[1] in ('E', 'S'):
                m_filters.append(-int(mode[2:]))
            else:
                try:
                    m_filters.append(int(mode[1:]))
                except ValueError:
                    raise click.UsageError(
                        f"Unknown direction/wavenumber in mode '{mode}'. Use W, E, or numbers.")

        periods_hours = sorted(list(set(periods_hours)), reverse=True)
        m_filters = sorted(list(set(m_filters)))
        logger.info(f"Parsed modes into periods: {periods_hours} and wavenumbers: {m_filters}")
    else:
        if not periods_str:
            raise click.UsageError("Must specify either --periods or --modes.")
        periods_hours = [float(p.strip()) for p in periods_str.split(',')]
        m_filters = [int(m.strip()) for m in m_str.split(',')] if m_str else None

    ds = _load_and_ensure_healpix(ifile)
    if var_name is None:
        var_name = _guess_variable(ds, "temperature")

    if method == 'ls':
        from .analysis import compute_leastsquares_tidal_analysis
        out_ds = compute_leastsquares_tidal_analysis(
            ds, var_name=var_name, periods_hours=periods_hours,
            m_filters=m_filters, lmax=lmax, time_dim=time_dim
        )
    elif method in ('sh', 'fourier'):
        from .analysis import compute_wavelet_tidal_analysis
        out_ds = compute_wavelet_tidal_analysis(
            ds, var_name=var_name, periods_hours=periods_hours,
            m_filters=m_filters, lmax=lmax, time_dim=time_dim,
            dj=dj, temporal_mean=temporal_mean, method=method,
            decompose_sym_asy=decompose_sym_asy,
        )
    else:
        raise click.UsageError(f"Unsupported method '{method}'.")

    logger.info(f"Computing and saving tidal analysis to {ofile}")
    recommended = out_ds.attrs.pop('_recommended_dask_scheduler', None)
    n_workers = int(out_ds.attrs.pop('_recommended_dask_num_workers', 1))
    import dask
    if recommended == 'synchronous' or n_workers == 1:
        logger.debug("Scheduler: synchronous (1 block at a time)")
        with dask.config.set(scheduler='synchronous'):
            write_dataset(out_ds, ofile)
    else:
        logger.debug(f"Scheduler: threads with {n_workers} worker(s)")
        with dask.config.set(scheduler='threads', num_workers=n_workers):
            write_dataset(out_ds, ofile)

    logger.info("Done.")


@cli.command()
@click.argument('ifile')
@click.option('--type', 'plot_type',
              type=click.Choice(['tides', 'section', 'map', 'spectrum', 'ep-flux']),
              required=True,
              help='Type of plot to generate.')
@click.option('--var', 'var_name', default=None,
              help='Variable to plot (for section, map, or spectrum).')
@click.option('--x-dim', default='lat', help='X dimension for section plots (default: lat)')
@click.option('--y-dim', default='z_mc', help='Y dimension for section plots (default: z_mc)')
@click.option('--height', 'target_height', type=float, default=None,
              help='Select level closest to this height (km) for map and spectrum plots. Avoids vertical averages.')
@click.option('--out-dir', default='.',
              help='Output directory for plots (default: current directory)')
@click.option('--prefix', default=None, help='Prefix for output filenames.')
@profile_command
def plot(ifile, plot_type, var_name, x_dim, y_dim, target_height, out_dir, prefix):
    """
    Generate simple visualizations for healicon products.

    Args:
        ifile: Path to input data file.
        plot_type: Type of plot to generate (tides, section, map, or spectrum).
        var_name: Variable to plot (for section, map, or spectrum).
        x_dim: X dimension for section plots (default: lat).
        y_dim: Y dimension for section plots (default: z_mc).
        target_height: Select level closest to this height (km) for map and spectrum plots.
        out_dir: Output directory for plots (default: current directory).
        prefix: Prefix for output filenames.
    """
    import xarray as xr
    import os
    from .visualize import plot_tides, plot_section, plot_map, plot_spectrum

    logger.info(f"Opening file: {ifile}")
    ds = xr.open_dataset(ifile, chunks='auto')

    if prefix is None:
        prefix = os.path.splitext(os.path.basename(ifile))[0]

    logger.info(f"Generating '{plot_type}' plot...")

    if plot_type == 'tides':
        plot_tides(ds, out_dir=out_dir, prefix=prefix)
    elif plot_type == 'section':
        if var_name is None:
            raise click.UsageError("Must specify --var for section plots.")
        plot_section(ds, var_name=var_name, x_dim=x_dim, y_dim=y_dim, out_dir=out_dir,
                     prefix=prefix)
    elif plot_type == 'map':
        if var_name is None:
            raise click.UsageError("Must specify --var for map plots.")
        plot_map(ds, var_name=var_name, target_height=target_height, out_dir=out_dir, prefix=prefix)
    elif plot_type == 'ep-flux':
        from .visualize import plot_ep_flux
        plot_ep_flux(ds, out_dir=out_dir, prefix=prefix)
    elif plot_type == 'spectrum':
        plot_spectrum(ds, var_name=var_name, target_height=target_height, out_dir=out_dir,
                      prefix=prefix)

    logger.info("Plotting complete.")


@cli.command('ep-flux')
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('--mode', type=click.Choice(['auto', 'full', 'qg']), default='auto', show_default=True,
              help='EP flux mode: full TEM, QG approximation, or auto (full when w is present).')
@click.option('--time-mean', is_flag=True, default=False,
              help='Average over the time dimension before saving.')
@profile_command
def ep_flux_cmd(ifile, ofile, mode, time_mean):
    """
    Compute Eliassen-Palm flux (F, ∇·F) and wave-induced acceleration.

    ifile: Input HEALPix dataset (must contain u and v; temp and pres recommended;
           w enables full TEM formulation).
    ofile: Output NetCDF file with F_phi, F_z, div_F, a_EP.
    """
    _check_io_safety(ifile, ofile)

    from .analysis.ep_flux import eliassen_palm

    ds = _load_and_ensure_healpix(ifile)
    logger.info(f"Computing Eliassen-Palm flux (mode={mode}) ...")
    out_ds = eliassen_palm(ds, mode=mode, time_mean=time_mean)
    write_dataset(out_ds, ofile)
    logger.info(f"EP flux saved to {ofile}.")


if __name__ == '__main__':
    cli()
