import logging

import click

from .core import run_sequential

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """HealICON: Interpolate atmospheric model outputs to HEALPix grid."""
    pass


def _load_and_ensure_healpix(input_file, target_nside=None):
    import xarray as xr
    import healpy as hp
    from .interpolate import interpolate_to_healpix

    logger.info(f"Opening file: {input_file}")
    ds = xr.open_dataset(input_file, chunks='auto')

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
        logger.info("Input dataset is not a HEALPix grid. Auto-interpolating first...")
        ds = interpolate_to_healpix(ds, nside=target_nside)

    return ds


@cli.command()
@click.option('-i', '--input', 'input_pattern', required=True,
              help='Input file pattern (e.g., "data/icon_*.nc"). Can include wildcards.')
@click.option('-o', '--output', 'output_template', required=True,
              help='Output file template (e.g., "output_{basename}"). '
                   '{basename} will be replaced by the input file name.')
@click.option('-n', '--nside', type=int, required=False, default=None,
              help='HEALPix Nside resolution parameter (e.g., 32, 64, 128). If omitted, defaults to closest resolution.')
@click.option('-c', '--config', 'config_path', type=click.Path(exists=True), default=None,
              help='Path to YAML configuration file for variable mapping.')
@click.option('-g', '--grid', 'grid_file', type=click.Path(exists=True), default=None,
              help='Optional path to external grid file containing coordinates (e.g., clat, clon).')
@click.option('--gpu', is_flag=True, default=False,
              help='Enable GPU acceleration for KDTree interpolation if available.')
def convert(input_pattern, output_template, nside, config_path, grid_file, gpu):
    """
    Convert model output to HEALPix grid sequentially.
    """
    run_sequential(
        input_pattern=input_pattern,
        output_template=output_template,
        nside=nside,
        config_path=config_path,
        grid_file=grid_file,
        use_gpu=gpu
    )


@cli.command()
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input HEALPix NetCDF file.')
@click.option('-o', '--output', 'output_file', required=True,
              help='Output NetCDF file.')
@click.option('-l', '--lat', type=float, required=True,
              help='Target latitude in degrees [-90, 90].')
@click.option('--num-lons', type=int, default=None,
              help='Number of longitude points to extract (default: number of HEALPix grid points).')
def extract_lat(input_file, output_file, lat, num_lons):
    """
    Extract data along all longitudes for a specific latitude from a HEALPix dataset.
    """
    from .extract import extract_along_latitude

    ds = _load_and_ensure_healpix(input_file)

    out_ds = extract_along_latitude(ds, lat=lat, num_lons=num_lons)

    logger.info(f"Computing and saving extracted data to {output_file}")
    out_ds.compute().to_netcdf(output_file)
    logger.info("Done.")


@cli.command()
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input HEALPix NetCDF file.')
@click.option('-o', '--output', 'output_file', required=True,
              help='Output NetCDF file.')
@click.option('-l', '--lon', type=float, required=True,
              help='Target longitude in degrees [-180, 180] or [0, 360].')
@click.option('--num-lats', type=int, default=None,
              help='Number of latitude points to extract.')
def extract_lon(input_file, output_file, lon, num_lats):
    """Extract data along all latitudes for a specific longitude."""
    from .extract import extract_along_longitude

    ds = _load_and_ensure_healpix(input_file)
    out_ds = extract_along_longitude(ds, lon=lon, num_lats=num_lats)
    logger.info(f"Computing and saving extracted data to {output_file}")
    out_ds.compute().to_netcdf(output_file)
    logger.info("Done.")


@cli.command()
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input HEALPix NetCDF file.')
@click.option('-o', '--output', 'output_file', required=True,
              help='Output NetCDF file.')
@click.option('--lat', type=float, required=True, help='Target latitude.')
@click.option('--lon', type=float, required=True, help='Target longitude.')
def extract_point(input_file, output_file, lat, lon):
    """Extract full time/height profile for a specific lat/lon point."""
    from .extract import extract_point as ep

    ds = _load_and_ensure_healpix(input_file)
    out_ds = ep(ds, lat=lat, lon=lon)
    logger.info(f"Computing and saving extracted data to {output_file}")
    out_ds.compute().to_netcdf(output_file)
    logger.info("Done.")


@cli.command()
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input HEALPix NetCDF file.')
@click.option('-o', '--output', 'output_file', required=True,
              help='Output NetCDF file.')
def zonal_mean(input_file, output_file):
    """Compute the zonal mean (longitude average) across HEALPix rings."""
    from .extract import zonal_mean as zm

    ds = _load_and_ensure_healpix(input_file)
    out_ds = zm(ds)
    logger.info(f"Computing and saving zonal mean to {output_file}")
    out_ds.compute().to_netcdf(output_file)
    logger.info("Done.")


@cli.command()
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input HEALPix NetCDF file.')
@click.option('-o', '--output', 'output_file', required=True,
              help='Output NetCDF file.')
@click.option('-v', '--var', 'var_name', required=True,
              help='Variable to compute power spectrum for.')
@click.option('--lmax', type=int, default=None,
              help='Maximum spherical harmonic degree l.')
def spectrum(input_file, output_file, var_name, lmax):
    """Compute the angular power spectrum (Cl) of a variable."""
    import healpy as hp
    from .analysis import compute_spectrum, degree_to_wavelength

    ds = _load_and_ensure_healpix(input_file)
    out_ds = compute_spectrum(ds, var_name=var_name, lmax=lmax)
    logger.info(f"Computing and saving spectrum to {output_file}")
    out_ds = out_ds.compute()
    # Log effective resolution after computing
    actual_lmax = int(out_ds['l'].max())
    nyquist_km = degree_to_wavelength(actual_lmax)
    logger.info(f"Spectrum resolved up to lmax={actual_lmax} (~{nyquist_km:.0f} km Nyquist scale).")
    out_ds.to_netcdf(output_file)
    logger.info("Done.")


@cli.command()
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input HEALPix NetCDF file.')
@click.option('-o', '--output', 'output_file', required=True,
              help='Output NetCDF file.')
@click.option('--fwhm', type=float, default=None,
              help='Full-width half-max in degrees for Gaussian smoothing.')
@click.option('--lmax', type=int, default=None,
              help='Hard low-pass spectral cutoff at spherical harmonic degree l.')
@click.option('--wavelength', 'wavelength_km', type=float, default=None,
              help='Hard low-pass spectral cutoff expressed as a physical wavelength in km.')
def filter(input_file, output_file, fwhm, lmax, wavelength_km):
    """Filter spatial maps using spherical harmonic transforms."""
    from .analysis import filter_spatial

    ds = _load_and_ensure_healpix(input_file)
    out_ds = filter_spatial(ds, fwhm_deg=fwhm, lmax=lmax, wavelength_km=wavelength_km)
    logger.info(f"Computing and saving filtered data to {output_file}")
    out_ds.compute().to_netcdf(output_file)
    logger.info("Done.")


@cli.command()
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input HEALPix NetCDF file.')
@click.option('-o', '--output', 'output_file', required=True,
              help='Output NetCDF file.')
@click.option('-n', '--nside', type=int, default=None,
              help='Target nside for the resolution change.')
@click.option('-z', '--zoom', type=int, default=None,
              help='Target zoom level (refinement), where nside = 2**zoom.')
def regrade(input_file, output_file, nside, zoom):
    """Upgrade or downgrade the HEALPix resolution."""
    from .analysis import regrade_resolution

    if nside is None and zoom is None:
        raise click.UsageError("You must provide either --nside or --zoom.")
    if zoom is not None:
        nside = 2 ** zoom

    ds = _load_and_ensure_healpix(input_file, target_nside=nside)

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
    logger.info(f"Computing and saving regraded data to {output_file}")
    out_ds.compute().to_netcdf(output_file)
    logger.info("Done.")


@cli.command()
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input HEALPix NetCDF file.')
@click.option('-o', '--output', 'output_file', required=True,
              help='Output NetCDF file.')
@click.option('--u', 'u_var', required=True, help='Name of eastward wind variable.')
@click.option('--v', 'v_var', required=True, help='Name of northward wind variable.')
@click.option('--lmax', type=int, default=None, help='Max spherical harmonic degree.')
def uv2dv(input_file, output_file, u_var, v_var, lmax):
    """Compute horizontal divergence and vorticity from U and V wind components."""
    from .analysis import compute_vorticity_divergence

    ds = _load_and_ensure_healpix(input_file)
    out_ds = compute_vorticity_divergence(ds, u_var=u_var, v_var=v_var, lmax=lmax)
    logger.info(f"Computing and saving vorticity/divergence to {output_file}")
    out_ds.compute().to_netcdf(output_file)
    logger.info("Done.")


@cli.command()
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input HEALPix NetCDF file.')
@click.option('-o', '--output', 'output_file', required=True,
              help='Output NetCDF file.')
@click.option('--div', 'div_var', required=True, help='Name of divergence variable.')
@click.option('--vor', 'vor_var', required=True, help='Name of vorticity variable.')
@click.option('--lmax', type=int, default=None, help='Max spherical harmonic degree.')
def dv2uv(input_file, output_file, div_var, vor_var, lmax):
    """Compute U and V wind components from horizontal divergence and vorticity."""
    from .analysis import compute_uv_from_vorticity_divergence

    ds = _load_and_ensure_healpix(input_file)
    out_ds = compute_uv_from_vorticity_divergence(ds, div_var=div_var, vor_var=vor_var, lmax=lmax)
    logger.info(f"Computing and saving U/V to {output_file}")
    out_ds.compute().to_netcdf(output_file)
    logger.info("Done.")


@cli.command()
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input file (HEALPix or native grid).')
@click.option('-o', '--output', 'output_file', required=True,
              help='Output NetCDF file.')
@click.option('--u', 'u_var', required=True, help='Name of eastward wind variable.')
@click.option('--v', 'v_var', required=True, help='Name of northward wind variable.')
@click.option('--lmax', type=int, default=None, help='Max spherical harmonic degree.')
@click.option('--psi/--no-psi', default=True, show_default=True,
              help='Include streamfunction ψ [m² s⁻¹] in output.')
@click.option('--chi/--no-chi', default=True, show_default=True,
              help='Include velocity potential χ [m² s⁻¹] in output.')
def helmholtz(input_file, output_file, u_var, v_var, lmax, psi, chi):
    """Helmholtz decomposition: split wind into rotational and divergent components.

    Outputs u_rot, v_rot (rotational wind) and u_div, v_div (divergent wind).
    Optionally also computes the streamfunction (--psi) and velocity potential
    (--chi), both in units of m² s⁻¹.
    """
    from .analysis import compute_helmholtz

    ds = _load_and_ensure_healpix(input_file)
    out_ds = compute_helmholtz(ds, u_var=u_var, v_var=v_var, lmax=lmax,
                                include_psi=psi, include_chi=chi)
    logger.info(f"Computing and saving Helmholtz decomposition to {output_file}")
    out_ds.compute().to_netcdf(output_file)
    logger.info("Done.")


if __name__ == '__main__':
    cli()
