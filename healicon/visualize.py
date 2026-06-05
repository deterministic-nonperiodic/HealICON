import os
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import healpy as hp

logger = logging.getLogger(__name__)

# Try to import cmasher, fallback to standard matplotlib cmaps
try:
    import cmasher as cmr
    wind_cm = cmr.fusion_r
except ImportError:
    wind_cm = 'RdYlBu_r'

# Simple temp colormap definition based on standard colors if file not present
# This mimics a standard diverging or sequential colormap
temp_cm = 'inferno'


def set_publication_style():
    """Apply publication-ready matplotlib parameters."""
    params = {
        'xtick.labelsize': 'small',
        'ytick.labelsize': 'small',
        'font.size': 13,
        'legend.title_fontsize': 13,
        'legend.fontsize': 13,
        'font.family': 'serif',
        'text.usetex': False,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'figure.titlesize': 16,
    }
    plt.rcParams.update(params)


def plot_tides(ds: xr.Dataset, out_dir: str = ".", prefix: str = "tides"):
    """
    Plot tidal amplitude and phase across latitudes and heights.
    Expects zonal mean dataset with 'lat' and 'z_mc' (or 'plev') coordinates.
    """
    set_publication_style()
    
    level_name = 'z_mc' if 'z_mc' in ds.coords else 'plev'
    
    if 'lat' in ds.coords and level_name in ds.coords:
        ds = ds.sortby(['lat', level_name])
    else:
        logger.warning("Dataset missing 'lat' or height coordinate for tidal cross-section.")
        
    p_12 = np.timedelta64(12, 'h')
    p_24 = np.timedelta64(24, 'h')

    modes = {
        'DW1': {'period': p_24, 'm': -1, 'type': 'Symmetric'},
        'SW2': {'period': p_12, 'm': -2, 'type': 'Symmetric'},
        'DE3': {'period': p_24, 'm': 3, 'type': 'Symmetric'},
        'SE2': {'period': p_12, 'm': 2, 'type': 'Symmetric'},
        'DW1_asy': {'period': p_24, 'm': -1, 'type': 'Antisymmetric'},
        'DE3_asy': {'period': p_24, 'm': 3, 'type': 'Antisymmetric'},
    }

    # Identify available modes in the dataset
    available_modes = {}
    for name, meta in modes.items():
        var_name = 'temp_amp_sym' if meta['type'] == 'Symmetric' else 'temp_amp_asy'
        if var_name in ds and 'period' in ds.coords and 'm' in ds.coords:
            try:
                # Check if exact period/m exists
                p_val = ds['period'].values
                m_val = ds['m'].values
                # We use nearest to just check if it's broadly there, or let sel handle it
                ds[var_name].sel(period=meta['period'], m=meta['m'], method='nearest')
                available_modes[name] = meta
            except Exception:
                pass
                
    if not available_modes:
        logger.error("No matching tidal modes found in dataset.")
        return

    n_modes = len(available_modes)
    n_cols = 2
    n_rows = int(np.ceil(n_modes / n_cols))

    # --- Amplitude Plot ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows + 1), sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()

    lat_ticks = [-60, -30, 0, 30, 60]
    lat_labels = ['60°S', '30°S', '0°', '30°N', '60°N']
    vmax = 6.0
    
    # Try to find max amplitude for better scaling
    try:
        max_amp = float(ds[['temp_amp_sym', 'temp_amp_asy']].to_array().max())
        if np.isfinite(max_amp) and max_amp > 0:
            vmax = min(max_amp, 20.0) # Cap at 20 for visibility
    except:
        pass
        
    levels = np.linspace(0, vmax, 13)

    cf = None
    for i, (name, meta) in enumerate(available_modes.items()):
        ax = axes[i]
        var_name = 'temp_amp_sym' if meta['type'] == 'Symmetric' else 'temp_amp_asy'
        data = ds[var_name].sel(period=meta['period'], m=meta['m'], method='nearest')
        
        y = data[level_name] / 1000.0 if level_name == 'z_mc' else data[level_name]
        x = data.lat
        
        cf = ax.contourf(x, y, data, levels=levels, cmap='inferno', extend='max')
        ax.set_title(f"{name.split('_')[0]} ({meta['type']})", fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        
    # Formatting
    for i, ax in enumerate(axes):
        if i >= n_modes:
            ax.remove()
        else:
            if i % n_cols == 0:
                ylabel = "Height / km" if level_name == 'z_mc' else "Pressure / hPa"
                ax.set_ylabel(ylabel)
            if i >= n_modes - n_cols:
                ax.set_xlabel("Latitude")
                ax.set_xlim(-60, 60)
                ax.set_xticks(lat_ticks)
                ax.set_xticklabels(lat_labels)
            ax.label_outer()
            
    if level_name == 'z_mc':
        axes[0].set_ylim(60, 110)
    elif level_name == 'plev':
        axes[0].invert_yaxis()

    fig.subplots_adjust(bottom=0.15, hspace=0.1, wspace=0.1)
    if cf:
        cbar_ax = fig.add_axes([0.26, 0.05, 0.52, 0.02])
        cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Amplitude / K')

    os.makedirs(out_dir, exist_ok=True)
    amp_out_path = os.path.join(out_dir, f"{prefix}_amplitude.png")
    plt.savefig(amp_out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved amplitude plot to {amp_out_path}")
    plt.close(fig)

    # --- Phase Plot ---
    fig_pha, axes_pha = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows + 1), sharex=True, sharey=True, squeeze=False)
    axes_pha = axes_pha.flatten()
    levels_pha = np.linspace(-np.pi, np.pi, 20)

    cf_pha = None
    for i, (name, meta) in enumerate(available_modes.items()):
        ax = axes_pha[i]
        var_name = 'temp_pha_sym' if meta['type'] == 'Symmetric' else 'temp_pha_asy'
        data = ds[var_name].sel(period=meta['period'], m=meta['m'], method='nearest')
        
        y = data[level_name] / 1000.0 if level_name == 'z_mc' else data[level_name]
        x = data.lat
        
        cf_pha = ax.contourf(x, y, data, levels=levels_pha, cmap='twilight_shifted', extend='both')
        ax.set_title(f"{name.split('_')[0]} Phase ({meta['type']})", fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)

    for i, ax in enumerate(axes_pha):
        if i >= n_modes:
            ax.remove()
        else:
            if i % n_cols == 0:
                ylabel = "Height / km" if level_name == 'z_mc' else "Pressure / hPa"
                ax.set_ylabel(ylabel)
            if i >= n_modes - n_cols:
                ax.set_xlabel("Latitude")
                ax.set_xlim(-60, 60)
                ax.set_xticks(lat_ticks)
                ax.set_xticklabels(lat_labels)
            ax.label_outer()

    if level_name == 'z_mc':
        axes_pha[0].set_ylim(60, 110)
    elif level_name == 'plev':
        axes_pha[0].invert_yaxis()

    fig_pha.subplots_adjust(bottom=0.15, hspace=0.1, wspace=0.1)
    if cf_pha:
        cbar_ax_pha = fig_pha.add_axes([0.26, 0.05, 0.52, 0.02])
        cbar_pha = fig_pha.colorbar(cf_pha, cax=cbar_ax_pha, orientation='horizontal', ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar_pha.ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        cbar_pha.set_label('Phase / rad')

    pha_out_path = os.path.join(out_dir, f"{prefix}_phase.png")
    plt.savefig(pha_out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved phase plot to {pha_out_path}")
    plt.close(fig_pha)


def plot_section(ds: xr.Dataset, var_name: str, x_dim: str = 'lat', y_dim: str = 'z_mc', out_dir: str = ".", prefix: str = "section"):
    """
    Plot a 2D cross-section (e.g. Latitude vs Height, or Time vs Height).
    """
    set_publication_style()
    
    if var_name not in ds:
        logger.error(f"Variable '{var_name}' not found in dataset.")
        return
        
    data = ds[var_name]
    
    # Squeeze out singleton dimensions
    data = data.squeeze()
    
    if x_dim not in data.coords or y_dim not in data.coords:
        logger.error(f"Dimensions '{x_dim}' or '{y_dim}' not available for plotting.")
        return

    # Average over remaining dimensions
    reduced_dims = [dim for dim in data.dims if dim not in [x_dim, y_dim]]
    if reduced_dims:
        logger.info(f"Averaging over additional dimensions: {reduced_dims}")
        data = data.mean(dim=reduced_dims)
        
    y_vals = data[y_dim] / 1000.0 if y_dim == 'z_mc' else data[y_dim]
    x_vals = data[x_dim]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Determine colormap based on variable
    cmap = temp_cm if 'temp' in var_name else wind_cm
    
    cf = ax.contourf(x_vals, y_vals, data, levels=20, cmap=cmap)
    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    
    units = ds[var_name].attrs.get('units', '')
    long_name = ds[var_name].attrs.get('long_name', var_name)
    cbar.set_label(f"{long_name} [{units}]" if units else long_name)
    
    # Axis labels
    ax.set_ylabel("Height / km" if y_dim == 'z_mc' else y_dim)
    ax.set_xlabel(x_dim.capitalize())
    
    if x_dim == 'lat':
        ax.set_xlim(-90, 90)
        ax.set_xticks([-60, -30, 0, 30, 60])
        ax.set_xticklabels(['60°S', '30°S', '0°', '30°N', '60°N'])
        
    if y_dim == 'plev':
        ax.invert_yaxis()
        ax.set_yscale('log')
        
    ax.set_title(f"{long_name} Cross-Section", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_{var_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved cross-section plot to {out_path}")
    plt.close(fig)


def plot_map(ds: xr.Dataset, var_name: str, target_height: float | None = None, out_dir: str = ".", prefix: str = "map"):
    """
    Plot a 2D global map of HEALPix or regular grid data.
    """
    set_publication_style()
    
    if var_name not in ds:
        logger.error(f"Variable '{var_name}' not found in dataset.")
        return
        
    data = ds[var_name].squeeze()
    
    # Check if HEALPix
    is_healpix = False
    from .grid import get_cells_dim
    try:
        cell_dim = get_cells_dim(ds)
        if cell_dim in data.dims:
            is_healpix = True
    except ValueError:
        pass
        
    title_suffix = ""
    # Handle vertical level selection
    vert_dims = [d for d in ['height', 'z_mc', 'altitude', 'plev'] if d in data.dims or d in ds.coords]
    if vert_dims:
        v_dim = vert_dims[0]
        max_val_global = ds[v_dim].max().item() if v_dim in ds else 1000
        is_meters = max_val_global > 500 and v_dim != 'plev'
        
        if target_height is not None:
            target_val = target_height * 1000.0 if is_meters else target_height
            logger.info(f"Selecting {v_dim} closest to {target_height} km (target val: {target_val}).")
            data = data.sel({v_dim: target_val}, method='nearest')
        elif v_dim in data.dims:
            logger.info(f"Selecting first level for {v_dim}.")
            data = data.isel({v_dim: 0})
            
        if v_dim in data.coords:
            val = data[v_dim].item()
            val_km = val / 1000.0 if is_meters else val
            title_suffix = f" at ~{val_km:.1f} km"

    reduced_dims = [dim for dim in data.dims if dim not in (cell_dim if is_healpix else ['lat', 'lon'])]
    if reduced_dims:
        logger.info(f"Selecting first index for extra dimensions: {reduced_dims}")
        data = data.isel({dim: 0 for dim in reduced_dims})
        
    units = ds[var_name].attrs.get('units', '')
    long_name = ds[var_name].attrs.get('long_name', var_name)
    cmap = temp_cm if 'temp' in var_name else wind_cm
    
    fig = plt.figure(figsize=(10, 5))
    
    if is_healpix:
        # HEALPix mollview
        # Healpy handles plotting differently, directly acting on current figure
        arr = data.values
        is_nested = ds.attrs.get('healpix_order', 'ring').lower() == 'nested'
        
        hp.mollview(arr, fig=fig.number, nest=is_nested, cmap=cmap, title=f"{long_name}{title_suffix}", unit=units)
    else:
        if 'lat' not in data.coords or 'lon' not in data.coords:
            logger.error("Dataset missing 'lat' or 'lon' coordinates for regular map.")
            return
            
        ax = plt.axes()
        cf = ax.contourf(data.lon, data.lat, data, levels=20, cmap=cmap)
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label(f"{long_name} [{units}]" if units else long_name)
        
        ax.set_title(f"{long_name} Map{title_suffix}", fontweight='bold')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle='--', alpha=0.5)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_{var_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved map plot to {out_path}")
    plt.close(fig)


def plot_spectrum(ds: xr.Dataset, var_name: str = None, target_height: float | None = None, out_dir: str = ".", prefix: str = "spectrum"):
    """
    Plot spectral energy or spherical harmonic power.
    """
    set_publication_style()
    
    # Find spectral variables
    if var_name:
        vars_to_plot = [var_name]
    else:
        vars_to_plot = [v for v in ds.data_vars if 'l' in ds[v].coords or 'wavenumber' in ds[v].coords]
        
    if not vars_to_plot:
        logger.error("No spectral variables found to plot.")
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for var in vars_to_plot:
        data = ds[var].squeeze()
        
        x_dim = 'l' if 'l' in data.dims else ('wavenumber' if 'wavenumber' in data.dims else None)
        if not x_dim:
            continue
            
        title_suffix = ""
        # Handle vertical level selection
        vert_dims = [d for d in ['height', 'z_mc', 'altitude', 'plev'] if d in data.dims or d in ds.coords]
        if vert_dims:
            v_dim = vert_dims[0]
            max_val_global = ds[v_dim].max().item() if v_dim in ds else 1000
            is_meters = max_val_global > 500 and v_dim != 'plev'
            
            if target_height is not None:
                target_val = target_height * 1000.0 if is_meters else target_height
                logger.info(f"Selecting {v_dim} closest to {target_height} km (target val: {target_val}).")
                data = data.sel({v_dim: target_val}, method='nearest')
            elif v_dim in data.dims:
                logger.info(f"Selecting first level for {v_dim}.")
                data = data.isel({v_dim: 0})
                
            if v_dim in data.coords:
                val = data[v_dim].item()
                val_km = val / 1000.0 if is_meters else val
                title_suffix = f" at ~{val_km:.1f} km"
                
        reduced_dims = [dim for dim in data.dims if dim != x_dim]
        if reduced_dims:
            logger.info(f"Selecting first index for extra dimensions: {reduced_dims}")
            data = data.isel({dim: 0 for dim in reduced_dims})
            
        x_vals = data[x_dim].values
        valid_idx = x_vals > 0
        
        # Get metadata
        var_base = var.replace('_cl', '')
        long_name = ds[var].attrs.get('long_name', ds.get(var_base, {}).attrs.get('long_name', var)) if hasattr(ds, var_base) else var
        units = ds[var].attrs.get('units', ds.get(var_base, {}).attrs.get('units', '')) if hasattr(ds, var_base) else ''
        label = f"{long_name} ({units})" if units else long_name
        
        ax.loglog(x_vals[valid_idx], data.values[valid_idx], label=label, linewidth=2)
        
    ax.set_xlabel("Spherical Harmonic Degree ($l$)", fontsize=12)
    ax.set_ylabel("Power / Energy", fontsize=12)
    
    # Use title_suffix from the last variable (assuming they share the same grid)
    full_title = f"Spectral Distribution{title_suffix}" if 'title_suffix' in locals() else "Spectral Distribution"
    ax.set_title(full_title, fontweight='bold', fontsize=14)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=11)
    
    # Add secondary axis for wavelength
    def l_to_wav(l):
        l = np.maximum(l, 1e-10)
        return (2 * np.pi * 6371.229) / l
        
    def wav_to_l(wav):
        wav = np.maximum(wav, 1e-10)
        return (2 * np.pi * 6371.229) / wav
        
    secax = ax.secondary_xaxis('top', functions=(l_to_wav, wav_to_l))
    secax.set_xlabel('Equivalent Wavelength (km)', fontsize=12)
    secax.set_xticks([10000, 5000, 2000, 1000, 500, 250, 100])
    secax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_spectrum.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved spectrum plot to {out_path}")
    plt.close(fig)
