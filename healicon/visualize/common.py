import logging
import re

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

try:
    import cmasher as cmr

    wind_cm = cmr.fusion_r
except ImportError:
    wind_cm = 'RdYlBu_r'

temp_cm = 'inferno'

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
