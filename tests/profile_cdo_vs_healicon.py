import time
import subprocess
import os

input_file = "examples/UA-ICON_NWP_temp_DOM01_HL_60-110km_202501.nc"
cdo_out = "cdo_out.nc"
healicon_out = "healicon_out.nc"
spectrum_out = "spectrum_out.nc"

print("====================================")
print("PROFILING HEALICON VS CDO")
print("====================================")

# 1. Profiling Spatial Conversion
print("\n[1] Spatial Conversion (ICON Unstructured -> Global Grid)")

# CDO Remapping to healpix nside=64
print("Running CDO remapbil,hpz6 (healpix nside=64)...")
start = time.time()
subprocess.run(["cdo", "-O", "remapbil,hpz6", input_file, cdo_out], capture_output=True)
cdo_time = time.time() - start
print(f"CDO remapbil took: {cdo_time:.2f} seconds")

# HealICON conversion to HEALPix nside=64 (comparable ~0.9 deg resolution)
print("Running healicon convert to HEALPix nside=64...")
start = time.time()
subprocess.run(["python3", "-m", "healicon.cli", "convert", input_file, healicon_out, "--nside", "64"], capture_output=True)
healicon_time = time.time() - start
print(f"HealICON convert took: {healicon_time:.2f} seconds")

speedup = cdo_time / healicon_time
print(f"-> HealICON is {speedup:.2f}x {'FASTER' if speedup > 1 else 'SLOWER'} than CDO interpolation.")

# 2. Profiling Spectral Analysis
print("\n[2] Spectral Analysis (Power Spectrum / Harmonics)")

# CDO Spectral decomposition (grid to spectral)
# Note: CDO gp2sp requires regular grid. We use the remapped cdo_out.
print("Running CDO gp2sp (Grid to Spectral)...")
start = time.time()
subprocess.run(["cdo", "-O", "gp2sp", cdo_out, "cdo_sp.nc"], capture_output=True)
cdo_spec_time = time.time() - start
print(f"CDO gp2sp took: {cdo_spec_time:.2f} seconds")

# HealICON Angular Power Spectrum (using hp.anafast)
print("Running healicon spectrum...")
start = time.time()
subprocess.run(["python3", "-m", "healicon.cli", "spectrum", healicon_out, spectrum_out], capture_output=True)
healicon_spec_time = time.time() - start
print(f"HealICON spectrum took: {healicon_spec_time:.2f} seconds")

speedup_spec = cdo_spec_time / healicon_spec_time
print(f"-> HealICON is {speedup_spec:.2f}x {'FASTER' if speedup_spec > 1 else 'SLOWER'} than CDO spectral transform.")

# Cleanup
for f in [cdo_out, healicon_out, spectrum_out, "cdo_sp.nc"]:
    if os.path.exists(f):
        os.remove(f)
