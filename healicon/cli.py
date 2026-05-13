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

if __name__ == '__main__':
    cli()
