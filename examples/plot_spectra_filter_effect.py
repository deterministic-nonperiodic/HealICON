import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

from healicon.interpolate import interpolate_to_healpix
from healicon.analysis import filter_spatial, compute_spectrum, wavelength_to_degree, EARTH_RADIUS_KM

def main():
    input_file = "UA-ICON_NWP_u_DOM01_ML_20250123T000000Z.nc"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found in current directory.")
        return

    print(f"Loading {input_file}...")
    ds_orig = xr.open_dataset(input_file)
    ds_subset = ds_orig[['u']].isel(time=0, height=0)

    # Use nside=256 to capture scales down to ~150km (Nyquist is lmax ~ 767 -> lambda ~ 50km)
    nside = 256
    print(f"Interpolating to HEALPix (nside={nside})...")
    ds_hp = interpolate_to_healpix(ds_subset, nside=nside).compute()

    scales_km = [1000, 500, 250]
    
    # 1. Compute spectrum of the ORIGINAL field
    print("Computing spectrum for unfiltered field...")
    ds_spc_orig = compute_spectrum(ds_hp, var_name='u').compute()
    
    spectra = {"Unfiltered": ds_spc_orig['u_cl'].values.flatten()}
    
    # 2. Filter at different scales and compute spectra
    for scale in scales_km:
        lmax_cutoff = int(wavelength_to_degree(scale))
        print(f"Filtering at {scale} km (lmax={lmax_cutoff})...")
        
        # Apply the hard spectral cutoff filter
        ds_filtered = filter_spatial(ds_hp, lmax=lmax_cutoff).compute()
        
        print(f"Computing spectrum for filtered field ({scale} km)...")
        # We compute the spectrum up to the same lmax as the original to see the drop-off
        # Actually, compute_spectrum will just compute up to 3*nside-1 automatically
        ds_spc_filt = compute_spectrum(ds_filtered, var_name='u').compute()
        
        spectra[f"Hard cutoff $\\lambda \\geq$ {scale} km (lmax={lmax_cutoff})"] = ds_spc_filt['u_cl'].values.flatten()

    # 2b. Gaussian filter at 1000 km FWHM for comparison
    gauss_scale_km = 1000.0
    # Convert km FWHM to degrees on the sphere
    fwhm_deg = np.rad2deg(gauss_scale_km / EARTH_RADIUS_KM)
    print(f"Applying Gaussian filter (FWHM={gauss_scale_km:.0f} km = {fwhm_deg:.2f} deg)...")
    ds_gauss = filter_spatial(ds_hp, fwhm_deg=fwhm_deg).compute()
    ds_spc_gauss = compute_spectrum(ds_gauss, var_name='u').compute()
    spectra[f"Gaussian FWHM {gauss_scale_km:.0f} km"] = ds_spc_gauss['u_cl'].values.flatten()
        
    # 3. Plotting
    print("Generating plot...")
    plt.figure(figsize=(9, 6))
    
    l_values = ds_spc_orig['l'].values
    
    # Plot original
    plt.loglog(l_values[1:], spectra["Unfiltered"][1:], label="Unfiltered", color='black', linewidth=2)
    
    colors = ['tab:red', 'tab:blue', 'tab:green']
    for idx, scale in enumerate(scales_km):
        key = list(spectra.keys())[idx + 1]  # skip 'Unfiltered'
        plt.loglog(l_values[1:], spectra[key][1:], label=key, color=colors[idx], alpha=0.8)

    # Gaussian filter line — dashed to distinguish from hard cutoffs
    gauss_key = list(spectra.keys())[-1]
    plt.loglog(l_values[1:], spectra[gauss_key][1:], label=gauss_key,
               color='tab:purple', linewidth=2, linestyle='--')

    # Noise floor: estimated from the mean of the top 10% of l in the unfiltered spectrum
    n_tail = max(1, len(l_values) // 10)
    noise_floor = np.mean(spectra['Unfiltered'][-n_tail:])
    ax1 = plt.gca()
    ax1.axhline(noise_floor, color='gray', linewidth=1.5, linestyle=':', label=f"Noise floor (~{noise_floor:.2e})")

    plt.title("Effect of Spectral Filtering on Kinetic Energy Spectrum ($u$ wind)", fontsize=14)
    plt.xlabel("Spherical Harmonic Degree (l)", fontsize=12)
    plt.ylabel("Angular Power Spectrum $C_l$", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    # Add a secondary axis for wavelength
    def l_to_wav(l):
        # avoid div by zero
        l = np.maximum(l, 1e-10)
        return (2 * np.pi * 6371.229) / l
        
    def wav_to_l(wav):
        wav = np.maximum(wav, 1e-10)
        return (2 * np.pi * 6371.229) / wav
        
    ax1 = plt.gca()
    secax = ax1.secondary_xaxis('top', functions=(l_to_wav, wav_to_l))
    secax.set_xlabel('Equivalent Wavelength (km)', fontsize=12)
    
    # Format the secondary axis nicely
    secax.set_xticks([10000, 5000, 2000, 1000, 500, 250, 100])
    secax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    save_path = "spectral_filtering_effect.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    main()
