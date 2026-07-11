# HealICON

HealICON is a command-line analysis tool for global atmospheric model outputs, utilizing HEALPix as the base grid to interpolate model data. It supports both regular (lon/lat) and unstructured grids (e.g. ICON), providing a seamless, equal-area environment optimized for analyzing atmospheric waves across the globe. Analysis include:

- Spectral analysis: angular power spectrum $C_l$, cross-spectra, kinetic energy spectra
- Spatial filtering: Gaussian beam, hard spectral cutoff, or wavenumber-specific
- Zonal averaging
- Vector calculus: vorticity/divergence, U/V winds to divergence/vorticity
- Tidal analysis: amplitude and phase, with optional wavenumber filtering and symmetric/antisymmetric decomposition
- Helmholtz decomposition: rotational/divergent components
- Extraction: vertical slices or 1D points
- Regridding: arbitrary nside or zoom-factor
- Eliassen-Palm flux: full primitive-equation TEM or quasi-geostrophic approximation on height or pressure coordinates; residual circulation (v*, w*) and wave-induced acceleration a_EP

## Features

- **Multi-Grid Support**: Interpolates from regular lon-lat grids or unstructured grids (e.g. ICON).
- **Sequential Processing**: Process large collections of model output files sequentially, avoiding memory overload, utilizing Dask for node-level parallelization.
- **Variable Mapping**: Flexibly select and rename variables via CF conventions automatically, or via a simple YAML namelist configuration.
- **CPU & GPU Acceleration**: Accelerated unstructured interpolation via SciPy `cKDTree` (CPU) or `cuml.NearestNeighbors` (GPU).
- **Auto-Resolution**: If the `nside` parameter is omitted, `HealICON` automatically computes the closest HEALPix resolution matching the original spatial grid size.
- **Analysis Suite**: Includes spherical harmonic spectral analysis, spatial filtering, zonal averaging, and vector calculus (vorticity/divergence).
- **Auto-Interpolation**: Analysis commands automatically detect raw model output (unstructured/regular) and seamlessly interpolate to HEALPix on the fly!

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
healicon convert -n 64 "path/to/icon_output.nc" "path/to/healpix_{basename}"
```

### Processing Multiple Files

Use wildcards in the input string to process multiple files sequentially:

```bash
healicon convert -n 64 "data/raw_*.nc" "output/hp_{basename}"
```
_Note: `{basename}` in the output template will automatically be replaced by the input file's name (e.g. `raw_01.nc`). You can also use `{name_no_ext}`._

**Command:**
```bash
healicon convert -n 64 -c config.yaml "data/icon_*.nc" "output/hp_{basename}"
```
If no config is provided, HealICON will attempt to use CF conventions (checking for the `standard_name` attribute) to select variables. If neither is found, all variables will be interpolated.

### External Grid File

For ICON output, the variables `clon` and `clat` (or `lon` and `lat`) are sometimes omitted from the model outputs and saved in a separate grid file. You can pass the external grid file via the `--grid` or `-g` parameter so that `HealICON` can successfully load the unstructured coordinates.

```bash
healicon convert -n 64 -g "data/icon_grid.nc" "data/icon_*.nc" "output/hp_{basename}"
```

### GPU Acceleration

If you have a GPU and `cuml` installed, you can enable GPU acceleration for the KDTree search when interpolating unstructured grids:

```bash
healicon convert -n 128 --gpu "data/icon_*.nc" "output/hp_{basename}"
```

### Analysis Suite

`HealICON` includes a powerful set of analysis tools that leverage `healpy` for spherical harmonic transforms. **Note:** If you pass a raw ICON native grid to any of these commands, it will automatically interpolate to HEALPix first!

```bash
# 1. Zonal Mean: Compute longitudinal averages over HEALPix latitude rings
healicon zonal-mean "data.nc" "zonal.nc"

# 2. Spectral Analysis: Compute angular power spectrum (C_l)
# Options include standard power spectrum, cross-spectra, and kinetic energy spectra (from u/v or div/vor).
healicon spectrum -v u --lmax 256 "data.nc" "spectrum.nc"
healicon spectrum -v temp -v q --type cross "data.nc" "cross_spectrum.nc"
healicon spectrum --type kinetic "data.nc" "kinetic_energy.nc"

# 3. Spatial Filtering: Apply a Gaussian beam (FWHM) or hard spectral cutoff (lmax)
healicon filter --fwhm 5.0 "data.nc" "filtered.nc"
healicon filter --lmax 15 "data.nc" "filtered.nc"

# 4. Vector Calculus: Convert between U/V winds and divergence/vorticity
healicon uv2dv --u u --v v "winds.nc" "kinematics.nc"
healicon dv2uv --div divergence --vor vorticity "kinematics.nc" "winds_recon.nc"

# 5. Helmholtz Decomposition: Split wind into rotational and divergent components
healicon helmholtz --u u --v v "winds.nc" "helmholtz.nc"

# 6. Extraction: Extract slices or points instantly
healicon extract-lat -l 45.0 "data.nc" "slice_lat.nc"
healicon extract-lon -l 180.0 "data.nc" "slice_lon.nc"
healicon extract-point --lat 45.0 --lon 10.0 "data.nc" "point.nc"

# 7. Tidal Analysis: Extract tidal amplitude and phase, with optional wavenumber filtering and symmetric/antisymmetric decomposition
healicon tides --modes DW1,SW2,DE3,SE2 "input_time_series.nc" "output_tides.nc"

# 8. Eliassen-Palm Flux: Compute F_phi, F_z, div F, and wave-induced acceleration a_EP
#    Requires u and v. Adding temp+pres enables QG and full TEM; adding w enables
#    the full primitive-equation TEM with Psi, v*, w* in the output.
#    When a relative vorticity field (vor/vorticity/zeta) is present, it is used
#    directly for f_hat = f + zeta_bar instead of finite-differencing u_zm.
healicon ep-flux "data.nc" "ep_flux.nc"                        # auto: full TEM if w present, else QG
healicon ep-flux --mode full "data.nc" "ep_flux.nc"            # force full TEM (requires w)
healicon ep-flux --mode qg   "data.nc" "ep_flux_qg.nc"         # quasi-geostrophic
healicon ep-flux --time-mean "data.nc" "ep_flux_mean.nc"       # time-average before saving

# 9. Regrade Resolution: Change HEALPix resolution using nside or zoom (nside=2^zoom)
healicon regrade --zoom 6 "data.nc" "regraded.nc"
```

### Visualization

`HealICON` includes a built-in plot command for quick inspection of analysis outputs:

```bash
# EP flux: quiver overlay of (F_phi, F_z) on a_EP background, with divergence contours
healicon plot --type ep-flux "ep_flux.nc"

# Cross-section (latitude x height) for any variable
healicon plot --type section --var temp "data.nc"

# Keogram (time x height), supports comma-separated variables for stacked panels
healicon plot --type keogram --var u        "data.nc"
healicon plot --type keogram --var u,v,temp "data.nc"

# Global HEALPix map at a target altitude
healicon plot --type map --var temp --height 85 "data.nc"

# Angular power spectrum
healicon plot --type spectrum --var u "data.nc"
```

### Python API

All analysis routines are accessible directly in Python:

```python
import xarray as xr
from healicon.analysis.ep_flux import eliassen_palm

ds = xr.open_dataset("data.nc")

# Full pipeline: eddy covariances -> F_phi, F_z, div_F, a_EP
ep_ds = eliassen_palm(ds, mode="auto")   # mode: 'auto' | 'full' | 'qg'

# Output variables (height-coordinate full TEM):
#   F_phi   [kg s-2]       EP flux, meridional component
#   F_z     [kg m-1 s-2]   EP flux, vertical component
#   div_F   [kg m-1 s-2]   EP flux divergence
#   a_EP    [m s-1 day-1]  wave-induced zonal acceleration
#   Psi     [kg s-1]       TEM mass stream function
#   v_star, w_star         residual-mean meridional / vertical velocity
#   u_zm, temp_zm, ...     zonal-mean diagnostics passed through
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
healicon convert -n 128 "/data/models/icon_output_*.nc" "/data/processed/hp_{basename}"
```
Dask will automatically detect the available CPUs via SLURM and utilize them.
