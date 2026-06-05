import xarray as xr
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np

ds = xr.open_dataset('../saber_healpix.nc')
data = ds['ktemp'].isel(altitude=200, ut=0).values.copy()
data[np.isnan(data)] = hp.UNSEEN

plt.figure(figsize=(10, 5))
hp.mollview(data, title="UT=0, Alt=200", nest=False)
plt.savefig('test_unfilled.png')
