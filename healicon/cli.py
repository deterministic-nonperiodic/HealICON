import click
import logging
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
    import xarray as xr
    from .extract import extract_along_latitude
    
    logger.info(f"Opening HEALPix file: {input_file}")
    ds = xr.open_dataset(input_file, chunks='auto')
    
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
    import xarray as xr
    from .extract import extract_along_longitude
    
    logger.info(f"Opening HEALPix file: {input_file}")
    ds = xr.open_dataset(input_file, chunks='auto')
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
    import xarray as xr
    from .extract import extract_point as ep
    
    logger.info(f"Opening HEALPix file: {input_file}")
    ds = xr.open_dataset(input_file, chunks='auto')
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
    import xarray as xr
    from .extract import zonal_mean as zm
    
    logger.info(f"Opening HEALPix file: {input_file}")
    ds = xr.open_dataset(input_file, chunks='auto')
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
    import xarray as xr
    from .analysis import compute_spectrum
    
    logger.info(f"Opening HEALPix file: {input_file}")
    ds = xr.open_dataset(input_file, chunks='auto')
    out_ds = compute_spectrum(ds, var_name=var_name, lmax=lmax)
    logger.info(f"Computing and saving spectrum to {output_file}")
    out_ds.compute().to_netcdf(output_file)
    logger.info("Done.")

@cli.command()
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input HEALPix NetCDF file.')
@click.option('-o', '--output', 'output_file', required=True,
              help='Output NetCDF file.')
@click.option('--fwhm', type=float, default=None,
              help='Full-width half-max in degrees for Gaussian smoothing.')
@click.option('--lmax', type=int, default=None,
              help='Hard low-pass spectral cutoff at degree l.')
def filter(input_file, output_file, fwhm, lmax):
    """Filter spatial maps using spherical harmonic transforms."""
    import xarray as xr
    from .analysis import filter_spatial
    
    logger.info(f"Opening HEALPix file: {input_file}")
    ds = xr.open_dataset(input_file, chunks='auto')
    out_ds = filter_spatial(ds, fwhm_deg=fwhm, lmax=lmax)
    logger.info(f"Computing and saving filtered data to {output_file}")
    out_ds.compute().to_netcdf(output_file)
    logger.info("Done.")

@cli.command()
@click.option('-i', '--input', 'input_file', required=True, type=click.Path(exists=True),
              help='Input HEALPix NetCDF file.')
@click.option('-o', '--output', 'output_file', required=True,
              help='Output NetCDF file.')
@click.option('-n', '--nside', type=int, required=True,
              help='Target nside for the resolution change.')
def regrade(input_file, output_file, nside):
    """Upgrade or downgrade the HEALPix resolution."""
    import xarray as xr
    from .analysis import regrade_resolution
    
    logger.info(f"Opening HEALPix file: {input_file}")
    ds = xr.open_dataset(input_file, chunks='auto')
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
def calc_vorticity(input_file, output_file, u_var, v_var, lmax):
    """Compute horizontal vorticity and divergence from U and V components."""
    import xarray as xr
    from .analysis import compute_vorticity_divergence
    
    logger.info(f"Opening HEALPix file: {input_file}")
    ds = xr.open_dataset(input_file, chunks='auto')
    out_ds = compute_vorticity_divergence(ds, u_var=u_var, v_var=v_var, lmax=lmax)
    logger.info(f"Computing and saving vorticity/divergence to {output_file}")
    out_ds.compute().to_netcdf(output_file)
    logger.info("Done.")

if __name__ == '__main__':
    cli()
