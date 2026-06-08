import logging
from typing import Dict, Any

import click

from .core import run_sequential

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress overly verbose healpy INFO logs
logging.getLogger('healpy').setLevel(logging.WARNING)

_CF_VARS_LOOKUP: Dict[str, Dict[str, Any]] = {
    "u": {"standard_names": {"eastward_wind"}, "units": {"m s-1", "m/s"}},
    "v": {"standard_names": {"northward_wind"}, "units": {"m s-1", "m/s"}},
    "w": {"standard_names": {"upward_air_velocity", "vertical_velocity_in_air"},
          "units": {"m s-1", "Pa s-1"}},
    "pressure": {"standard_names": {"air_pressure"}, "units": {"Pa", "pascal"}},
    "temperature": {"standard_names": {"air_temperature"}, "units": {"K", "kelvin"}},
    "density": {"standard_names": {"air_density"}, "units": {"kg / m**3", "kg m-3"}},
    "theta": {"standard_names": {"air_potential_temperature"}, "units": {"K", "kelvin"}},
    "divergence": {"standard_names": {"divergence_of_wind"}, "units": {"s-1"}},
    "vorticity": {"standard_names": {"relative_vorticity"}, "units": {"s-1"}},
}


def _cf_guess(ds, target: str) -> str | None:
    """
    Very light CF-based guess for a logical variable name.

    Looks at ``standard_name`` and common units to suggest a candidate
    when a configured variable is missing. Advisory only.
    """
    rule = _CF_VARS_LOOKUP.get(target)
    if rule is None:
        return None
    for name, da in ds.data_vars.items():
        std = str(da.attrs.get("standard_name", "")).strip()
        units = str(da.attrs.get("units", "")).strip()
        if std in rule["standard_names"] or any(u in units for u in rule["units"]):
            return name
    return None


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
    pass


def _load_and_ensure_healpix(ifile, target_nside=None):
    import xarray as xr
    import healpy as hp
    from .interpolate import interpolate_to_healpix

    logger.info(f"Opening file: {ifile}")
    ds = xr.open_dataset(ifile, chunks='auto')

    # Optimize Dask chunking: Spatial dimensions must be fully contiguous for spectral analysis
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
        ds = ds.chunk({dim: -1 for dim in spatial_dims}).unify_chunks()

    is_healpix = False
    try:
        from .grid import get_cells_dim
        cell_dim = get_cells_dim(ds)
        npix = ds.sizes[cell_dim]
        try:
            detected_nside = hp.npix2nside(npix)
            if hp.isnsideok(detected_nside):
                is_healpix = True
        except Exception:
            pass
    except ValueError:
        pass

    if ds.attrs.get('healpix_scheme') == 'RING':
        is_healpix = True

    for var in ds.data_vars:
        if ds[var].attrs.get('grid_mapping') == 'healpix':
            is_healpix = True
            break

    if not is_healpix:
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
    import os
    if os.path.abspath(ifile) == os.path.abspath(ofile):
        raise click.UsageError(
            "Input and output files cannot be the same. This would corrupt the input file.")

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
def extract_lat(ifile, ofile, lat, num_lons):
    """
    Extract data along all longitudes for a specific latitude from a HEALPix dataset.
    """
    import os
    if os.path.abspath(ifile) == os.path.abspath(ofile):
        raise click.UsageError(
            "Input and output files cannot be the same. This would corrupt the input file.")

    from .extract import extract_along_latitude

    ds = _load_and_ensure_healpix(ifile)

    out_ds = extract_along_latitude(ds, lat=lat, num_lons=num_lons)

    logger.info(f"Computing and saving extracted data to {ofile}")
    out_ds.compute().to_netcdf(ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('-l', '--lon', type=float, required=True,
              help='Target longitude in degrees [-180, 180] or [0, 360].')
@click.option('--num-lats', type=int, default=None,
              help='Number of latitude points to extract.')
def extract_lon(ifile, ofile, lon, num_lats):
    """Extract data along all latitudes for a specific longitude."""
    import os
    if os.path.abspath(ifile) == os.path.abspath(ofile):
        raise click.UsageError(
            "Input and output files cannot be the same. This would corrupt the input file.")

    from .extract import extract_along_longitude

    ds = _load_and_ensure_healpix(ifile)
    out_ds = extract_along_longitude(ds, lon=lon, num_lats=num_lats)
    logger.info(f"Computing and saving extracted data to {ofile}")
    out_ds.compute().to_netcdf(ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('--lat', type=float, required=True, help='Target latitude.')
@click.option('--lon', type=float, required=True, help='Target longitude.')
def extract_point(ifile, ofile, lat, lon):
    """Extract full time/height profile for a specific lat/lon point."""
    import os
    if os.path.abspath(ifile) == os.path.abspath(ofile):
        raise click.UsageError(
            "Input and output files cannot be the same. This would corrupt the input file.")

    from .extract import extract_point as ep

    ds = _load_and_ensure_healpix(ifile)
    out_ds = ep(ds, lat=lat, lon=lon)
    logger.info(f"Computing and saving extracted data to {ofile}")
    out_ds.compute().to_netcdf(ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('--spatial-dim', type=str, default='cells',
              help='Name of the spatial dimension (default: cells).')
@click.option('--time-dim', type=str, default=None,
              help='Optional name of the time/LST dimension to interpolate across temporally.')
def fill(ifile, ofile, spatial_dim, time_dim):
    """
    Fill missing values (NaNs) natively using HEALPix KDTree spatial nearest-neighbor and temporal 1D linear interpolation.
    """
    import os
    if os.path.abspath(ifile) == os.path.abspath(ofile):
        raise click.UsageError(
            "Input and output files cannot be the same. This would corrupt the input file.")

    from .curation import fill_healpix_gaps
    import xarray as xr

    # Load dataset
    ds = xr.open_dataset(ifile, chunks='auto')

    out_ds = fill_healpix_gaps(ds, spatial_dim=spatial_dim, time_dim=time_dim)

    logger.info(f"Saving filled dataset to {ofile}")
    out_ds.compute().to_netcdf(ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
def zonal_mean(ifile, ofile):
    """Compute the zonal mean (longitude average) across HEALPix rings."""
    import os
    if os.path.abspath(ifile) == os.path.abspath(ofile):
        raise click.UsageError(
            "Input and output files cannot be the same. This would corrupt the input file.")

    from .extract import zonal_mean as zm

    ds = _load_and_ensure_healpix(ifile)
    out_ds = zm(ds)
    logger.info(f"Computing and saving zonal mean to {ofile}")
    out_ds.compute().to_netcdf(ofile)
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
def spectrum(ifile, ofile, var_name, lmax, spectrum_type):
    """Compute the angular spectrum (Cl) of a variable or pair of variables."""
    import os
    if os.path.abspath(ifile) == os.path.abspath(ofile):
        raise click.UsageError(
            "Input and output files cannot be the same. This would corrupt the input file.")

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
    out_ds.to_netcdf(ofile)
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
def filter(ifile, ofile, fwhm, lmax, wavelength_km):
    """Filter spatial maps using spherical harmonic transforms."""
    import os
    if os.path.abspath(ifile) == os.path.abspath(ofile):
        raise click.UsageError(
            "Input and output files cannot be the same. This would corrupt the input file.")

    from .analysis import filter_spatial

    ds = _load_and_ensure_healpix(ifile)
    out_ds = filter_spatial(ds, fwhm_deg=fwhm, lmax=lmax, wavelength_km=wavelength_km)
    logger.info(f"Computing and saving filtered data to {ofile}")
    out_ds.compute().to_netcdf(ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('-n', '--nside', type=int, default=None,
              help='Target nside for the resolution change.')
@click.option('-z', '--zoom', type=int, default=None,
              help='Target zoom level (refinement), where nside = 2**zoom.')
def regrade(ifile, ofile, nside, zoom):
    """Upgrade or downgrade the HEALPix resolution."""
    import os
    if os.path.abspath(ifile) == os.path.abspath(ofile):
        raise click.UsageError(
            "Input and output files cannot be the same. This would corrupt the input file.")

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
    out_ds.compute().to_netcdf(ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('--u', 'u_var', default=None, help='Name of eastward wind variable.')
@click.option('--v', 'v_var', default=None, help='Name of northward wind variable.')
@click.option('--lmax', type=int, default=None, help='Max spherical harmonic degree.')
def uv2dv(ifile, ofile, u_var, v_var, lmax):
    """Compute horizontal divergence and vorticity from U and V wind components."""
    import os
    if os.path.abspath(ifile) == os.path.abspath(ofile):
        raise click.UsageError(
            "Input and output files cannot be the same. This would corrupt the input file.")

    from .analysis import compute_vorticity_divergence

    ds = _load_and_ensure_healpix(ifile)
    if u_var is None:
        u_var = _guess_variable(ds, "u")
    if v_var is None:
        v_var = _guess_variable(ds, "v")
    out_ds = compute_vorticity_divergence(ds, u_var=u_var, v_var=v_var, lmax=lmax)
    logger.info(f"Computing and saving vorticity/divergence to {ofile}")
    out_ds.compute().to_netcdf(ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('--div', 'div_var', default=None, help='Name of divergence variable.')
@click.option('--vor', 'vor_var', default=None, help='Name of vorticity variable.')
@click.option('--lmax', type=int, default=None, help='Max spherical harmonic degree.')
def dv2uv(ifile, ofile, div_var, vor_var, lmax):
    """Compute U and V wind components from horizontal divergence and vorticity."""
    import os
    if os.path.abspath(ifile) == os.path.abspath(ofile):
        raise click.UsageError(
            "Input and output files cannot be the same. This would corrupt the input file.")

    from .analysis import compute_uv_from_vorticity_divergence

    ds = _load_and_ensure_healpix(ifile)
    if div_var is None:
        div_var = _guess_variable(ds, "divergence")
    if vor_var is None:
        vor_var = _guess_variable(ds, "vorticity")
    out_ds = compute_uv_from_vorticity_divergence(ds, div_var=div_var, vor_var=vor_var, lmax=lmax)
    logger.info(f"Computing and saving U/V to {ofile}")
    out_ds.compute().to_netcdf(ofile)
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
def helmholtz(ifile, ofile, u_var, v_var, lmax, psi, chi):
    """Helmholtz decomposition: split wind into rotational and divergent components.

    Outputs u_rot, v_rot (rotational wind) and u_div, v_div (divergent wind).
    Optionally also computes the streamfunction (--psi) and velocity potential
    (--chi), both in units of m² s⁻¹.
    """
    import os
    if os.path.abspath(ifile) == os.path.abspath(ofile):
        raise click.UsageError(
            "Input and output files cannot be the same. This would corrupt the input file.")

    from .analysis import compute_helmholtz

    ds = _load_and_ensure_healpix(ifile)
    if u_var is None:
        u_var = _guess_variable(ds, "u")
    if v_var is None:
        v_var = _guess_variable(ds, "v")
    out_ds = compute_helmholtz(ds, u_var=u_var, v_var=v_var, lmax=lmax,
                               include_psi=psi, include_chi=chi)
    logger.info(f"Computing and saving Helmholtz decomposition to {ofile}")
    out_ds.compute().to_netcdf(ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile', type=click.Path(exists=True))
@click.argument('ofile')
@click.option('-v', '--var', 'var_name', default=None,
              help='Variable to compute tidal analysis for.')
@click.option('-p', '--periods', 'periods_str', default=None,
              help='Comma-separated tidal periods in hours. E.g., 12.0,24.0 for semidiurnal and diurnal.')
@click.option('-m', '--m-filters', 'm_str', default=None,
              help='Optional comma-separated zonal wavenumbers. E.g., 1,2,3.')
@click.option('--modes', 'modes_str', default=None,
              help='Comma-separated tidal modes. E.g., DW1, SW2, DE3, SE2. '
                   'If provided, automatically configures periods and m-filters.')
@click.option('--lmax', type=int, default=None, help='Max spherical harmonic degree.')
@click.option('--time-dim', default='time', show_default=True,
              help='Dimension name for time-like axis (e.g., "time" or "lst").')
def tides(ifile, ofile, var_name, periods_str, m_str, modes_str, lmax, time_dim):
    """
    Perform a full tidal analysis (temporal fit and spatial symmetry decomposition).
    
    Extracts the amplitude and phase of given periods (in hours), optionally 
    filters to specific zonal wavenumbers (-m), and decomposes the resulting 
    spatial patterns into symmetric and antisymmetric components relative to the equator.
    """
    import os
    if os.path.abspath(ifile) == os.path.abspath(ofile):
        raise click.UsageError(
            "Input and output files cannot be the same. This would corrupt the input file.")

    from .analysis import compute_tidal_analysis

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
                m_filters.append(-int(mode[2:]))
            elif mode[1] in ('E', 'S'):
                m_filters.append(int(mode[2:]))
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
    out_ds = compute_tidal_analysis(ds, var_name=var_name, periods_hours=periods_hours,
                                    m_filters=m_filters, lmax=lmax, time_dim=time_dim)
    logger.info(f"Computing and saving tidal analysis to {ofile}")
    out_ds.compute().to_netcdf(ofile)
    logger.info("Done.")


@cli.command()
@click.argument('ifile')
@click.option('--type', 'plot_type', type=click.Choice(['tides', 'section', 'map', 'spectrum']),
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
def plot(ifile, plot_type, var_name, x_dim, y_dim, target_height, out_dir, prefix):
    """Generate simple visualizations for healicon products."""
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
    elif plot_type == 'spectrum':
        plot_spectrum(ds, var_name=var_name, target_height=target_height, out_dir=out_dir,
                      prefix=prefix)

    logger.info("Plotting complete.")


if __name__ == '__main__':
    cli()
