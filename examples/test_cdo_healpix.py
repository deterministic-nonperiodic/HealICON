import xarray as xr
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from healicon.analysis import compute_spectrum

def main():
    # 1. Generate some dummy data on a HEALPix grid (nside=16)
    nside = 16
    npix = hp.nside2npix(nside)
    lmax = 3 * nside - 1
    
    # Generate some random power spectrum and then a map from it
    cl = 1.0 / (np.arange(lmax + 1) + 1)**2
    cl[0] = 0
    map_data = hp.synfast(cl, nside, lmax=lmax)
    
    # 2. Create an xarray Dataset that mimics CDO HEALPix output
    # CDO uses 'cell' or 'cells' for the spatial dimension
    ds = xr.Dataset(
        data_vars={
            'temp': (('cell',), map_data)
        },
        coords={
            'cell': np.arange(npix)
        }
    )
    
    # Add CDO-compliant HEALPix grid mapping
    ds['healpix'] = np.int32(1)
    ds['healpix'].attrs = {
        'grid_mapping_name': 'healpix',
        'healpix_nside': nside,
        'healpix_order': 'ring'
    }
    
    # Link the data variable to the grid mapping
    ds['temp'].attrs['grid_mapping'] = 'healpix'
    
    print("Created dummy CDO-style HEALPix dataset:")
    print(ds)
    
    # 3. Compute spectrum using our function
    print("\nComputing spectrum...")
    # Because compute_spectrum checks for 'cell' and the grid mapping variable, it should just work!
    ds_spc = compute_spectrum(ds, var_name='temp', lmax=lmax)
    
    # 4. Plot the result to verify
    plt.figure(figsize=(8, 5))
    plt.loglog(ds_spc['l'].values[1:], ds_spc['temp_cl'].values[1:], 'b-', label='Recovered Spectrum')
    plt.loglog(np.arange(1, lmax + 1), cl[1:], 'r--', label='Input Spectrum')
    plt.title('Spectrum of CDO-style HEALPix Data')
    plt.xlabel('Spherical Harmonic Degree (l)')
    plt.ylabel('Power')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    out_file = 'cdo_spectrum_test.png'
    plt.savefig(out_file)
    print(f"\nSaved plot to {out_file}")

if __name__ == '__main__':
    main()
