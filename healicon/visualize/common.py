import logging
import re

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

try:
    import cmasher as cmr

    wind_cm = cmr.fusion_r
except ImportError:
    wind_cm = 'RdYlBu_r'

_TEMP_CM_COLORS = [
    '#3e0214', '#3d0216', '#570b25', '#6e1531', '#87203e',
    '#9f294c', '#af4d4c', '#be704c', '#c38a53', '#c19d61',
    '#c2ab75', '#aba87d', '#879a84', '#648d89', '#648d89',
    '#438190', '#287593', '#27678a', '#275b80', '#254f77',
    '#26436f', '#2f4775', '#39517f', '#415c87', '#4d6591',
    '#56719c', '#617ba5', '#7591b9', '#7f9bc3', '#8aa4cd',
    '#93b1d7', '#9cb8df', '#a7bfe3', '#afc6e6', '#b8cdea',
    '#c0d4ed', '#cbdbf4', '#d3e2f7', '#dce9fa', '#e4f0ff',
    '#e7f8ff',
]
temp_cm = LinearSegmentedColormap.from_list('temp_c42', _TEMP_CM_COLORS[::-1])

VARIABLE_ATTRS: dict[str, dict] = {
    'temp': {
        'label': 'Temperature', 'units': 'K',
        'factor': 1.0, 'v_range': [160., 282., 20.], 'colormap': temp_cm,
    },
    'u': {
        'label': 'Zonal velocity', 'units': r'm s$^{-1}$',
        'factor': 1.0, 'v_range': [-90., 91., 20.], 'colormap': wind_cm,
    },
    'v': {
        'label': 'Meridional velocity', 'units': r'm s$^{-1}$',
        'factor': 1.0, 'v_range': [-60., 61., 20.], 'colormap': wind_cm,
    },
    'w': {
        'label': 'Vertical velocity', 'units': r'cm s$^{-1}$',
        'factor': 1e2, 'v_range': [-300., 301., 50.], 'colormap': wind_cm,
    },
    'theta': {
        'label': 'Potential temperature', 'units': 'K',
        'factor': 1.0, 'v_range': [200., 1000., 100.], 'colormap': temp_cm,
    },
    'tke': {
        'label': 'Turbulent kinetic energy', 'units': r'm$^2$ s$^{-2}$',
        'factor': 1.0, 'v_range': [0., 20., 5.], 'colormap': temp_cm,
    },
}

SPECTRAL_KEYMAP: dict[str, str] = {
    'kinetic_energy_cl': r'$E_K$',
    'u_cl': r'$C_l^{u}$', 'v_cl': r'$C_l^{v}$', 'w_cl': r'$C_l^{w}$',
    'ua_cl': r'$C_l^{u}$', 'va_cl': r'$C_l^{v}$', 'wa_cl': r'$C_l^{w}$',
    'temperature_cl': r'$C_l^{T}$', 'temp_cl': r'$C_l^{T}$', 'T_cl': r'$C_l^{T}$',
    'ta_cl': r'$C_l^{T}$',
    'theta_cl': r'$C_l^{\theta}$', 'pt_cl': r'$C_l^{\theta}$',
    'qv_cl': r'$C_l^{q_v}$', 'hus_cl': r'$C_l^{q_v}$', 'q_cl': r'$C_l^{q}$',
    'divergence_cl': r'$C_l^{D}$', 'div_cl': r'$C_l^{D}$',
    'vorticity_cl': r'$C_l^{\zeta}$', 'vor_cl': r'$C_l^{\zeta}$', 'zeta_cl': r'$C_l^{\zeta}$',
    'zg_cl': r'$C_l^{\Phi}$', 'geopot_cl': r'$C_l^{\Phi}$', 'phi_cl': r'$C_l^{\Phi}$',
    'pres_cl': r'$C_l^{p}$', 'ps_cl': r'$C_l^{p_s}$',
    'rho_cl': r'$C_l^{\rho}$',
}


def cf_to_latex(unit_string: str) -> str:
    _ABBREV = {
        'kelvin': 'K', 'meter': 'm', 'second': 's', 'kilogram': 'kg',
        'pascal': 'Pa', 'joule': 'J', 'watt': 'W', 'radian': 'rad',
        'degree': 'deg', 'kilometer': 'km',
    }
    for long, short in _ABBREV.items():
        unit_string = re.sub(rf'\b{long}\b', short, unit_string, flags=re.IGNORECASE)
    unit_string = unit_string.replace('**', '^')
    unit_string = re.sub(r'\s*\^\s*', '^', unit_string)

    def wrap_exponent(match):
        base, exp = match.groups()
        return f"{base}^{{{exp}}}"

    unit_string = re.sub(r'([A-Za-z]+)\^(-?\d+(?:\.\d+)?)', wrap_exponent, unit_string)
    unit_string = re.sub(r'(?<!\^)(-?\d+)(?=\s|$)', r'^{\1}', unit_string)
    unit_string = re.sub(r'\s+', r'\\,', unit_string.strip())
    return f"${unit_string}$" if unit_string else ""


def set_publication_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.title_fontsize': 11,
        'figure.titlesize': 14,
        'figure.titleweight': 'bold',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'grid.color': 'gray',
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.minor.width': 0.8,
        'ytick.minor.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })
