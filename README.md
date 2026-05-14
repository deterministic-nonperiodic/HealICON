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

# 4. Vector Calculus: Compute vorticity and divergence perfectly from U/V
healicon calc-vorticity -i "winds.nc" -o "vort.nc" --u u --v v

# 5. Extraction: Extract slices or points instantly
healicon extract-lat -i "data.nc" -o "slice_lat.nc" -l 45.0
healicon extract-lon -i "data.nc" -o "slice_lon.nc" -l 180.0
healicon extract-point -i "data.nc" -o "point.nc" --lat 45.0 --lon 10.0

# 6. Regrade Resolution: Change HEALPix resolution using nside or zoom (nside=2^zoom)
healicon regrade -i "data.nc" -o "regraded.nc" --zoom 6
```

## Output Metadata

`HealICON` preserves the global metadata from your input files and injects specific attributes describing the HEALPix transformation:

- `healpix_nside`: The resolution parameter of the HEALPix grid.
- `healpix_npix`: The total number of pixels on the sphere.
- `healpix_scheme`: Always set to **`RING`**. In the RING scheme, pixels are numbered continuously along lines of constant latitude (rings). This is highly optimized for computing spherical harmonics and latitudinal slices.
- `healpix_cell_area_sr`: The area of a single grid cell in **steradians** (`sr`). Because HEALPix is an equal-area grid, every cell has the exact same area: 4π / N_pix. *(To calculate the area in square meters, multiply this value by the square of your planetary radius).*

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
