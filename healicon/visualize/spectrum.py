import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .common import SPECTRAL_KEYMAP, cf_to_latex, set_publication_style
from ..cf_coords import _find_coordinate, _coord_is_meter

logger = logging.getLogger(__name__)


def _add_reference_slopes(ax, l_arr, data_arr):
    """
    Overlay canonical atmospheric kinetic-energy reference slopes on a log-log spectral plot.

    Two slopes are drawn:
      • l⁻³   anchored in the synoptic range  (l ≤ 20)
      • l⁻⁵·³ anchored in the mesoscale range (20 < l ≤ min(lmax, 1000))

    The magnitude of each reference line is set so that it passes through the median
    of the plotted data inside the corresponding l-range, keeping the lines informative
    without obscuring the actual spectra.

    Args:
        ax:      Matplotlib axes object (must already be in log-log mode).
        l_arr:   1-D array of spherical harmonic degrees (positive integers, l ≥ 1).
        data_arr: 1-D array of power/energy values aligned with l_arr.
    """

    l_arr = np.asarray(l_arr, dtype=float)
    data_arr = np.asarray(data_arr, dtype=float)

    slopes = [
        (-3, (2, 20), r"$l^{-3}$"),
        (-5 / 3, (20, 1000), r"$l^{-5/3}$"),
    ]

    for exp, (l_lo, l_hi), label in slopes:
        l_hi = min(l_hi, l_arr[-1])
        if l_lo >= l_hi:
            continue

        # Build a dense l-grid for the reference line
        l_ref = np.geomspace(l_lo, l_hi, 120)

        # Anchor magnitude: median of the data in this l-range (ignore NaN)
        mask = (l_arr >= l_lo) & (l_arr <= l_hi)
        if mask.sum() == 0 or not np.any(np.isfinite(data_arr[mask])):
            continue
        l_anchor = np.sqrt(l_lo * l_hi)  # geometric-mean anchor point
        d_median = np.nanmedian(data_arr[mask])
        amplitude = 10 * d_median / (l_anchor ** exp)  # so ref(l_anchor) == d_median

        y_ref = amplitude * l_ref ** exp

        ax.plot(l_ref, y_ref, lw=1.2, ls='--', color='gray', zorder=1)

        # Label near the peak (highest-y) end of the slope line
        ax.annotate(
            label,
            xy=(l_ref[0], y_ref[0]),
            xytext=(-4, 4),
            textcoords='offset points',
            color='dimgray',
            fontsize=13,
            ha='right',
            va='bottom',
        )

    # Draw a faint vertical line separating the two slope regimes
    if l_arr[-1] > 20:
        ax.axvline(x=20, color='gray', lw=0.8, ls=':', alpha=0.5, zorder=0)


def plot_spectrum(ds: xr.Dataset, var_name: str = None, target_height: float | None = None,
                  out_dir: str = ".", prefix: str = "spectrum"):
    """
    Plot spectral energy or spherical harmonic power spectrum.

    Produces a publication-quality log-log plot styled after spectra_base_figure, with:
      - Constrained layout
      - Degree ticks [1, 10, 100, 1000] via ScalarFormatter (minor ticks suppressed)
      - Y-tick labels left-aligned with padding (seba style)
      - AnchoredText box for the plot title in the upper-right corner
      - Dual x-axis: spherical harmonic degree (bottom), equivalent wavelength km (top)
      - Reference slope lines l⁻³ (synoptic, l ≤ 20) and l⁻⁵·³ (mesoscale, 20 < l ≤ 1000),
        anchored to the median of the plotted data in each range

    Args:
        ds: Input xarray Dataset containing spectral variables (dimension 'l' or 'wavenumber').
        var_name: Name of the variable to plot (optional; defaults to all spectral variables).
        target_height: Vertical level to select, in km (optional).
        out_dir: Output directory for the PNG file.
        prefix: Filename prefix for the output PNG.
    """
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.ticker import ScalarFormatter, NullFormatter

    set_publication_style()

    # Find spectral variables
    if var_name:
        vars_to_plot = [var_name]
    else:
        vars_to_plot = [v for v in ds.data_vars if
                        'l' in ds[v].coords or 'wavenumber' in ds[v].coords]

    if not vars_to_plot:
        logger.error("No spectral variables found to plot.")
        return

    # --- Build figure with constrained_layout (seba style) ---
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # --- Degree-axis tick formatting: show [1, 10, 100, 1000], hide minor labels ---
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())

    # Collect all plotted (l, value) pairs for reference-slope anchoring
    all_l = []
    all_vals = []
    title_suffix = ""
    lmin_observed = None  # set from first variable
    lmax_observed = 0  # updated from every variable; stays at data truncation
    ydata_min = np.inf  # finite positive extremes across all variables
    ydata_max = -np.inf
    all_units = []  # units string per variable (in plot order)
    all_long_names = []  # long_name per variable

    for var in vars_to_plot:
        data = ds[var].squeeze()

        x_dim = 'l' if 'l' in data.dims else ('wavenumber' if 'wavenumber' in data.dims else None)
        if not x_dim:
            continue

        # Handle vertical level selection
        # _find_coordinate expects a Dataset; wrap the DataArray temporarily.
        # Scoping to *data* (not ds) prevents picking up vertical coords that
        # belong to other variables and are not dims of this one.
        level_coord = _find_coordinate(data.to_dataset(name=var), 'level', raise_notfound=False)
        if level_coord is not None:
            v_dim = level_coord.name
            is_meters = _coord_is_meter(level_coord)

            if target_height is not None and v_dim in data.dims:
                target_val = target_height * 1000.0 if is_meters else target_height
                logger.info(
                    f"Selecting {v_dim} closest to {target_height} km (target val: {target_val}).")
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

        # Look up a concise display name; fall back to the short_name stripped of '_cl'.
        # The SPECTRAL_KEYMAP is also used for the y-axis label (single-variable case).
        attrs = ds[var].attrs
        long_name = attrs.get('long_name', attrs.get('standard_name', var))
        units = attrs.get('units', '')
        short_name = var.replace('_cl', '')
        display_name = SPECTRAL_KEYMAP.get(var, short_name)

        all_long_names.append(long_name)
        all_units.append(units)

        # Keep only l > 0 (l=0 is the global mean)
        data_valid = data.sel({x_dim: data[x_dim] > 0})

        l_vals = data_valid[x_dim].values.astype(float)
        v_vals = data_valid.values.ravel().astype(float)

        ax.plot(l_vals, v_vals, lw=1.8, label=display_name)

        # Accumulate for slope anchoring (first variable drives the reference)
        if not len(all_l):
            all_l = l_vals
            all_vals = v_vals

        if lmin_observed is None:
            lmin_observed = int(l_vals[0])
        lmax_observed = max(lmax_observed, int(l_vals[-1]))

        # Track finite positive extremes for y-limit snapping
        finite_pos = v_vals[np.isfinite(v_vals) & (v_vals > 0)]
        if finite_pos.size:
            ydata_min = min(ydata_min, finite_pos.min())
            ydata_max = max(ydata_max, finite_pos.max())

    # --- x-axis limits and major ticks (seba style) ---
    if len(all_l) and lmax_observed > 0:
        l_min = max(1, lmin_observed if lmin_observed is not None else 1)
        l_max = lmax_observed + 50
        ax.set_xlim(l_min, l_max)

        # Tick positions derived from the actual data range – no hard-coded 1000
        candidate_ticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
        x_ticks = candidate_ticks[(candidate_ticks >= l_min) & (candidate_ticks <= l_max)]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)

    # --- Reference slopes ---
    if len(all_l) > 0:
        _add_reference_slopes(ax, all_l, all_vals)

    # --- Y-limits: one decade above/below data extremes ---
    if np.isfinite(ydata_min) and np.isfinite(ydata_max) and ydata_min > 0:
        y_lo = 10 ** (np.floor(np.log10(ydata_min)) - 1)
        y_hi = 10 ** (np.ceil(np.log10(ydata_max)) + 1)
        ax.set_ylim(y_lo, y_hi)

    # --- Y-tick labels: left-aligned with extra pad (seba style) ---
    for tick_label in ax.yaxis.get_ticklabels():
        tick_label.set_horizontalalignment('left')
    ax.yaxis.set_tick_params(pad=30)

    # --- Y-label, title and legend (single vs. multi-variable) ---
    _fs = 12
    is_single = (len(vars_to_plot) == 1)

    if is_single:
        # Descriptive y-label: "<long_name> / <units in LaTeX>"
        _ln = all_long_names[0] if all_long_names else vars_to_plot[0]
        _u = all_units[0] if all_units else ''
        y_label = f"{_ln} / {cf_to_latex(_u)}" if _u else _ln

        # AnchoredText: "$E_K$ at ~50.0 km" (math symbol + altitude)
        _solo_var = vars_to_plot[0]
        _solo_display = SPECTRAL_KEYMAP.get(_solo_var, _solo_var.replace('_cl', ''))
        if title_suffix:
            at = AnchoredText(
                f"{_solo_display} {title_suffix.strip()}", prop=dict(size=_fs - 1),
                frameon=True, loc='upper right',
            )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
    else:
        # Generic y-label with the list of unique units
        unique_units = list(dict.fromkeys(u for u in all_units if u))
        units_str = ', '.join(cf_to_latex(u) for u in unique_units) if unique_units else ''
        y_label = f"Power spectra / {units_str}" if units_str else "Power spectra"

        # Legend with math-symbol labels; altitude as legend title
        legend_title = title_suffix.strip() if title_suffix else None
        ax.legend(fontsize=11, framealpha=0.9, title=legend_title,
                  title_fontsize=11)

    ax.set_xlabel("Spherical harmonic degree $l$", fontsize=_fs, labelpad=3)
    ax.set_ylabel(y_label, fontsize=_fs)

    # --- Secondary x-axis: wavelength in km (top, seba style) ---
    _R_KM = 6371.229

    def _l_to_wav(l):
        return (2 * np.pi * _R_KM) / np.maximum(l, 1e-10)

    def _wav_to_l(wav):
        return (2 * np.pi * _R_KM) / np.maximum(wav, 1e-10)

    secax = ax.secondary_xaxis('top', functions=(_l_to_wav, _wav_to_l))
    secax.xaxis.set_major_formatter(ScalarFormatter())
    secax.set_xlabel(r'wavelength / km', fontsize=_fs, labelpad=6)

    # Choose wavelength ticks that correspond to degrees within the plotted range
    wav_candidates = np.array([20000, 10000, 5000, 2000, 1000, 500, 250, 100, 50])
    if len(all_l):
        wav_at_lmin = _l_to_wav(all_l[0])
        wav_at_lmax = _l_to_wav(all_l[-1])
        wav_ticks = wav_candidates[
            (wav_candidates <= wav_at_lmin) & (wav_candidates >= wav_at_lmax)
            ]
        if len(wav_ticks):
            secax.set_xticks(wav_ticks)

    os.makedirs(out_dir, exist_ok=True)
    height_tag = f"_{target_height:.1f}km" if target_height is not None else ""
    out_path = os.path.join(out_dir, f"{prefix}{height_tag}_spectrum.png")

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved spectrum plot to {out_path}")
    plt.close(fig)
