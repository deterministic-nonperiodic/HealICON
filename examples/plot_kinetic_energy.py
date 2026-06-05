"""
Example: Computing Kinetic Energy Spectrum

This script demonstrates how to compute the kinetic energy spectrum of the atmosphere 
using two different but mathematically equivalent approaches in HealICON:
1. Directly from the U and V wind components using a spin-1 vector spherical harmonic transform.
2. From Divergence and Vorticity using standard scalar spherical harmonic transforms.

Both methods yield the exact same physical kinetic energy spectrum!
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

from healicon.interpolate import interpolate_to_healpix
from healicon.analysis import compute_spectrum, compute_vorticity_divergence

def main():
    # 1. Load data
    ifile = "UA-ICON_NWP_atm_DOM01_ML_20250202T000000Z.nc"
    if not os.path.exists(ifile):
        print(f"File not found: {ifile}")
        print("Please ensure you are running this script from the examples/ directory")
        return
        
    print(f"Loading {ifile}...")
    ds = xr.open_dataset(ifile).isel(time=0, height=-1)  # Surface level only for speed
    
    # Interpolate to HEALPix if it's on the native grid
    print("Interpolating to HEALPix (nside=64)...")
    ds_hp = interpolate_to_healpix(ds, nside=64).compute()
    
    lmax = 128
    
    # 2. Compute Divergence and Vorticity
    print("Computing Divergence and Vorticity from u and v...")
    ds_dv = compute_vorticity_divergence(ds_hp, u_var='u', v_var='v', lmax=lmax).compute()
    
    # 3. Compute KE Spectrum from U and V directly
    print("Computing KE Spectrum from U and V (spin-1 transform)...")
    spc_uv = compute_spectrum(ds_hp, var_name=['u', 'v'], lmax=lmax, spectrum_type='kinetic').compute()
    
    # 4. Compute KE Spectrum from Divergence and Vorticity
    print("Computing KE Spectrum from Divergence and Vorticity (scalar transform)...")
    spc_dv = compute_spectrum(ds_dv, var_name=['divergence', 'vorticity'], lmax=lmax, spectrum_type='kinetic').compute()
    
    # 5. Plotting
    print("Plotting comparison...")
    l = spc_uv['l'].values
    cl_uv = spc_uv['kinetic_energy_cl'].values
    cl_dv = spc_dv['kinetic_energy_cl'].values
    
    plt.figure(figsize=(10, 6))
    # We drop the l=0 (mean) component for log-log plotting
    plt.loglog(l[1:], cl_uv[1:], 'b-', linewidth=4, alpha=0.6, label='KE from $u, v$ (spin-1)')
    plt.loglog(l[1:], cl_dv[1:], 'r--', linewidth=2, label='KE from $div, vor$ (scalar)')
    
    plt.title('Kinetic Energy Spectrum Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Spherical Harmonic Degree ($l$)', fontsize=12)
    plt.ylabel('Kinetic Energy Density ($m^2 s^{-2}$)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    out_file = 'kinetic_energy_comparison.png'
    plt.savefig(out_file, dpi=150)
    print(f"Saved plot to '{out_file}'")
    
    ratio = np.nanmean(cl_dv[1:] / cl_uv[1:])
    print(f"Mean ratio between the two spectra: {ratio:.12f}")

if __name__ == "__main__":
    main()
