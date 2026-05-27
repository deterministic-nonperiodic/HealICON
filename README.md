# HealICON

HealICON is a command-line Python tool that efficiently interpolates atmospheric model outputs to the HEALPix grid using `xarray`, `healpy`, and `dask`. It supports both regular latitude-longitude grids and unstructured grids like the ICON icosahedral grid.

## Features

- **Multi-Grid Support**: Automatically detects and interpolates from regular lon-lat grids or unstructured grids (e.g. ICON).
- **Sequential Processing**: Process large collections of model output files sequentially, avoiding memory overload, utilizing Dask for node-level parallelization.
- **Variable Mapping**: Flexibly select and rename variables via CF conventions automatically, or via a simple YAML namelist configuration.
- **CPU & GPU Acceleration**: Accelerated unstructured interpolation via SciPy `cKDTree` (CPU) or `cuml.NearestNeighbors` (GPU).
- **Auto-Resolution**: If the `nside` parameter is omitted, `HealICON` automatically computes the closest HEALPix resolution matching the original spatial grid size.
- **Analysis Suite**: Includes spherical harmonic spectral analysis, spatial filtering, zonal averaging, and vector calculus (vorticity/divergence).
- **Auto-Interpolation**: Analysis commands automatically detect raw model output (unstructured/regular) and seamlessly interpolate to HEALPix on the fly!
- **SLURM Compatible**: Works perfectly within an `sbatch` job.

## Installation

To install via Conda (recommended):
```bash
conda env create -f environment.yml
conda activate healicon
pip install -e .
```

You can also install `HealICON` directly from GitHub using `pip`:

```bash
pip install git+https://github.com/deterministic-nonperiodic/HealICON.git
```

To install with GPU support (requires NVIDIA RAPIDS `cuml` and `cupy` installed in your environment):
```bash
pip install "HealICON[gpu] @ git+https://github.com/deterministic-nonperiodic/HealICON.git"
```

### Manual Installation (For Developers)

If you want to modify the source code, clone the repository and install it in editable mode:

```bash
git clone https://github.com/deterministic-nonperiodic/HealICON.git
cd HealICON
pip install -e .
```

To include the testing suite and development dependencies:
```bash
pip install -e .[test]
```

## Usage

### Basic Usage

Interpolate a single file to a HEALPix grid with `nside=64`. If `-n` is omitted, the resolution will be inferred automatically:

```bash
healicon convert -i "path/to/icon_output.nc" -o "path/to/healpix_{basename}" -n 64
```

### Processing Multiple Files

Use wildcards in the input string to process multiple files sequentially:

```bash
healicon convert -i "data/raw_*.nc" -o "output/hp_{basename}" -n 64
```
_Note: `{basename}` in the output template will automatically be replaced by the input file's name (e.g. `raw_01.nc`). You can also use `{name_no_ext}`._

### Variable Mapping (Namelist)

You can specify a YAML file to explicitly map input variables to output variables.

**config.yaml**
```yaml
variables:
  # output_name: input_name
  temp: t
  u_wind: u
```

**Command:**
```bash
healicon convert -i "data/icon_*.nc" -o "output/hp_{basename}" -n 64 -c config.yaml
```
If no config is provided, HealICON will attempt to use CF conventions (checking for the `standard_name` attribute) to select variables. If neither is found, all variables will be interpolated.

### External Grid File

For ICON output, the variables `clon` and `clat` (or `lon` and `lat`) are sometimes omitted from the model outputs and saved in a separate grid file. You can pass the external grid file via the `--grid` or `-g` parameter so that `HealICON` can successfully load the unstructured coordinates.

```bash
healicon convert -i "data/icon_*.nc" -o "output/hp_{basename}" -n 64 -g "data/icon_grid.nc"
```

### GPU Acceleration

If you have a GPU and `cuml` installed, you can enable GPU acceleration for the KDTree search when interpolating unstructured grids:

```bash
healicon convert -i "data/icon_*.nc" -o "output/hp_{basename}" -n 128 --gpu
```

### Analysis Suite

`HealICON` includes a powerful set of analysis tools that leverage `healpy` for spherical harmonic transforms. **Note:** If you pass a raw ICON native grid to any of these commands, it will automatically interpolate to HEALPix first!

```bash
# 1. Zonal Mean: Compute longitudinal averages over HEALPix latitude rings
healicon zonal-mean -i "data.nc" -o "zonal.nc"

# 2. Spectral Analysis: Compute angular power spectrum (C_l)
healicon spectrum -i "data.nc" -o "spectrum.nc" -v u --lmax 256

# 3. Spatial Filtering: Apply a Gaussian beam (FWHM) or hard spectral cutoff (lmax)
healicon filter -i "data.nc" -o "filtered.nc" --fwhm 5.0
healicon filter -i "data.nc" -o "filtered.nc" --lmax 15

# 4. Vector Calculus: Convert between U/V winds and divergence/vorticity
healicon uv2dv -i "winds.nc" -o "kinematics.nc" --u u --v v
healicon dv2uv -i "kinematics.nc" -o "winds_recon.nc" --div divergence --vor vorticity

# 5. Helmholtz Decomposition: Split wind into rotational and divergent components
healicon helmholtz -i "winds.nc" -o "helmholtz.nc" --u u --v v

# 6. Extraction: Extract slices or points instantly
healicon extract-lat -i "data.nc" -o "slice_lat.nc" -l 45.0
healicon extract-lon -i "data.nc" -o "slice_lon.nc" -l 180.0
healicon extract-point -i "data.nc" -o "point.nc" --lat 45.0 --lon 10.0

# 7. Regrade Resolution: Change HEALPix resolution using nside or zoom (nside=2^zoom)
healicon regrade -i "data.nc" -o "regraded.nc" --zoom 6
```

## Output Metadata

`HealICON` preserves the global metadata from your input files and injects CDO-compliant variables describing the HEALPix grid:

- **Spatial Dimension**: The grid points are flattened into a 1D spatial dimension named `cells`.
- **Coordinates**: `lat` and `lon` are standard data variables (not dimension coordinates), avoiding strict 1D coordinate matching issues in `xarray`.
- **Grid Mapping Variable**: A dummy integer variable named `healpix` is added, containing standard CF grid mapping attributes (`grid_mapping_name`, `healpix_nside`, `healpix_order`). All data variables are linked to it via `grid_mapping = "healpix"`.

HealICON interpolations natively output in the `RING` HEALPix ordering, but analysis tools will automatically detect and adapt to `NESTED` data if processing CDO-generated files.

## SLURM Example

To run `HealICON` on a compute node using SLURM, simply write an `sbatch` script:

```bash
#!/bin/bash
#SBATCH --job-name=healicon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=healicon_%j.log

# Load your python environment
source activate your_env

# Run healicon
healicon convert -i "/data/models/icon_output_*.nc" -o "/data/processed/hp_{basename}" -n 128
```
Dask will automatically detect the available CPUs via SLURM and utilize them.
