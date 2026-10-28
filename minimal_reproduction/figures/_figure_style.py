"""Shared PRB figure settings and output paths."""
from pathlib import Path
import os
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

DPI = 600
# Single-panel figures use a common physical canvas; colourbars fit inside it.
SINGLE_PANEL_SIZE = (3.6, 2.65)
# Temperature is encoded consistently in all temperature-comparison figures.
TEMPERATURE_CMAP = LinearSegmentedColormap.from_list(
    'prb_temperature', [(0.0, '#33539E'), (0.50, '#637EB5'),
                        (0.57, '#8877B1'), (0.70, '#AD628F'), (1.0, '#87446B')], N=1024)
TEMPERATURE_NORM = Normalize(vmin=300, vmax=800)

def temperature_color(temperature):
    return TEMPERATURE_CMAP(TEMPERATURE_NORM(temperature))

def apply_style():
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8.0, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
        'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2,
        'axes.grid': False, 'axes.linewidth': 0.7,
        'axes.spines.top': True, 'axes.spines.right': True,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
        'xtick.major.width': 0.65, 'ytick.major.width': 0.65,
        'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
        'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
        'xtick.major.pad': 3, 'ytick.major.pad': 3,
        'legend.fontsize': 7.2, 'legend.title_fontsize': 7.2,
        'legend.fancybox': False, 'legend.framealpha': 0.94,
        'legend.edgecolor': '#D3D6DA',
        'text.color': '#262626', 'axes.labelcolor': '#262626',
        'axes.edgecolor': '#333333', 'xtick.color': '#333333', 'ytick.color': '#333333',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.dpi': DPI, 'savefig.facecolor': 'white',
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.06,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    })

def participation_cmap():
    # Sequential lightness; avoids the nearly white middle of the old palette.
    return mpl.colors.ListedColormap(mpl.colormaps['viridis'](np.linspace(0.15, 0.85, 256)))


def output_path(name):
    root = Path(__file__).resolve().parents[2]
    default = root / 'minimal_reproduction' / 'outputs' / 'figures'
    path = Path(os.environ.get('GETE_FIGURE_OUTPUT', str(default))).resolve()
    allowed = root / 'minimal_reproduction' / 'outputs'
    if path == root or (path.is_relative_to(root) and not path.is_relative_to(allowed)):
        raise ValueError('Generated figures must not overwrite archived figures or inputs')
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def save_figure(fig, filename, **kwargs):
    """Save the PNG and optional editable PDF/SVG with the same canvas settings."""
    fig.savefig(filename, **kwargs)
    formats = os.environ.get('GETE_FIGURE_FORMATS', 'png').split(',')
    vector_kwargs = {key: value for key, value in kwargs.items() if key != 'metadata'}
    if 'pdf' in formats:
        fig.savefig(Path(filename).with_suffix('.pdf'), **vector_kwargs)
    if 'svg' in formats:
        fig.savefig(Path(filename).with_suffix('.svg'), **vector_kwargs)
