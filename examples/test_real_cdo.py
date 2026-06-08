import xarray as xr
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from healicon.interpolate import interpolate_to_healpix
from healicon.analysis import compute_spectrum
from healicon.grid import create_healpix_dataset
import os

def main():
    ifile = "UA-ICON_NWP_atm_DOM01_ML_20250202T000000Z.nc"
    if not os.path.exists(ifile):
        print(f"File {ifile} not found.")
        return

    print("1. Extracting subset (time=0, level=-1, u-wind, v-wind) to speed up testing...")
    ds = xr.open_dataset(ifile)
    # Extract u and v wind at surface
    ds_uv = ds[['u', 'v']].isel(time=0, height=-1)
    ds_uv.to_netcdf("uv_surface.nc")
    
    print("2. Generating target HEALPix grid description file for CDO...")
    import healpy as hp
    nside = 64
    npix = hp.nside2npix(nside)
    lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    
    with open("target_healpix_64.txt", "w") as f:
        f.write("gridtype = unstructured\n")
        f.write(f"gridsize = {npix}\n")
        f.write("xvals = " + " ".join(map(str, lon)) + "\n")
        f.write("yvals = " + " ".join(map(str, lat)) + "\n")
    
    import time
    
    print("3. Running CDO remapnn and scalar spectrum calculation...")
    cdo_start_time = time.time()
    # cdo remapnn,target_grid infile outfile
    cdo_cmd = ["cdo", "-remapnn,target_healpix_64.txt", "uv_surface.nc", "uv_cdo_healpix.nc"]
    try:
        subprocess.run(cdo_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"CDO failed: {e.stderr.decode()}")
        return
        
    ds_cdo = xr.open_dataset("uv_cdo_healpix.nc")
    if 'healpix' not in ds_cdo:
        ds_cdo['healpix'] = np.int32(1)
        ds_cdo['healpix'].attrs = {
            'grid_mapping_name': 'healpix',
            'healpix_nside': 64,
            'healpix_order': 'ring'
        }
        ds_cdo['u'].attrs['grid_mapping'] = 'healpix'
        ds_cdo['v'].attrs['grid_mapping'] = 'healpix'
        
    spc_u_cdo = compute_spectrum(ds_cdo, var_name='u', lmax=128)
    spc_v_cdo = compute_spectrum(ds_cdo, var_name='v', lmax=128)
    
    # CDO Naive Kinetic Energy: 0.5 * (u_cl + v_cl)
    ke_cdo_cl = 0.5 * (spc_u_cdo['u_cl'] + spc_v_cdo['v_cl'])
    ke_cdo_cl.compute()
    cdo_time = time.time() - cdo_start_time
    print(f"CDO Pipeline Time: {cdo_time:.2f} seconds")
    
    print("5. Calculating true kinetic energy spectrum directly from file with healicon...")
    healicon_start_time = time.time()
    ds_hp_healicon = interpolate_to_healpix(ds_uv, nside=64)
    # True Kinetic Energy spectrum using Spin-1 transform
    spc_healicon = compute_spectrum(ds_hp_healicon, var_name=['u', 'v'], spectrum_type='kinetic', lmax=128)
    spc_healicon.compute()
    healicon_time = time.time() - healicon_start_time
    print(f"HealICON Pipeline Time: {healicon_time:.2f} seconds")
    
    print("6. Plotting comparison...")
    plt.figure(figsize=(10, 6))
    l_cdo = spc_u_cdo['l'].values[1:]
    cl_cdo = ke_cdo_cl.values[1:]
    
    l_healicon = spc_healicon['l'].values[1:]
    cl_healicon = spc_healicon['kinetic_energy_cl'].values[1:]
    
    plt.loglog(l_healicon, cl_healicon, 'b-', linewidth=3, alpha=0.7, label=f'HealICON (True KE Spin-1, {healicon_time:.2f}s)')
    plt.loglog(l_cdo, cl_cdo, 'r--', linewidth=2, label=f'CDO approach (0.5*(u_cl+v_cl), {cdo_time:.2f}s)')
    
    plt.title('Kinetic Energy Spectrum: True (HealICON) vs Naive Scalar Sum (CDO)')
    plt.xlabel('Spherical Harmonic Degree (l)')
    plt.ylabel('Kinetic Energy Density')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    out_file = 'real_data_cdo_vs_healicon.png'
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")

if __name__ == '__main__':
    main()
