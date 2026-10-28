#!/usr/bin/env python3
"""
fig_s01_hnemd_convergence.py
Fig S1 — HNEMD running-average diagnostic from kappa.out.
It displays the evolution of the reported conductivity estimator over the
10 ns production segment; it is not an HCACF integral.
"""
from _figure_style import output_path, save_figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _figure_style import (apply_style, temperature_color, TEMPERATURE_CMAP,
                           TEMPERATURE_NORM, participation_cmap)
import seaborn as sns
import numpy as np
import os

# === Paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import native_kappa_path
OUTPUT = str(output_path('fig_s01_hnemd_convergence.png'))
OUTPUT_STEM = os.path.splitext(OUTPUT)[0]

# === Parameters ===
TEMPS_PLOT = [300, 585, 800]
OUTPUT_INTERVAL = 1000
TIME_STEP_FS = 1.0

# === Color palette (project standard) ===
C = {
    'navy':     '#6FA8C9',
    'sky':      '#A9CFE5',
    'lavender': '#B5A2D4',
    'pink':     '#E8B5C8',
    'mauve':    '#9F91C7',
    'grey':     '#78838F',
}

# === Styling ===
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.15)
apply_style()

TEMP_STYLES = {
    300: {'color': temperature_color(300), 'marker': 'o', 'ls': '-'},
    585: {'color': temperature_color(585), 'marker': 's', 'ls': '--'},
    800: {'color': temperature_color(800), 'marker': '^', 'ls': ':'},
}


def load_hnemd_estimator(temp, direction):
    """Load kappa.out, return (time_ns, running_avg, final_kappa, final_err)."""
    kappa_path = native_kappa_path(temp, direction)
    if not os.path.exists(kappa_path):
        raise FileNotFoundError(kappa_path)

    data = np.loadtxt(kappa_path)
    if data.ndim == 1:
        data = data.reshape(-1, 5)

    # Para: Fe || z → col 4; Perp: Fe || x → col 0 + col 1
    if direction == 'para':
        k_inst = data[:, 4]
    else:
        k_inst = data[:, 0] + data[:, 1]

    n = len(k_inst)
    time_ns = np.arange(1, n + 1) * OUTPUT_INTERVAL * TIME_STEP_FS * 1e-6
    running_avg = np.cumsum(k_inst) / np.arange(1, n + 1)

    # Block-averaged standard error
    n_block = min(10, n // 2)
    block_size = n // n_block
    block_means = np.array([
        np.mean(k_inst[i * block_size:(i + 1) * block_size])
        for i in range(n_block)
    ])
    std_err = np.std(block_means, ddof=1) / np.sqrt(n_block)

    return time_ns, running_avg, running_avg[-1], std_err


# === Main: 2-panel figure ===
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

for ax_idx, (direction, label, title_label) in enumerate([
    ('para', r'$\kappa_\parallel$', '(a)'),
    ('perp', r'$\kappa_\perp$',   '(b)'),
]):
    ax = axes[ax_idx]
    for temp in TEMPS_PLOT:
        result = load_hnemd_estimator(temp, direction)
        if result is None:
            continue
        time_ns, running_avg, final_k, final_err = result
        style = TEMP_STYLES[temp]

        ax.plot(time_ns, running_avg, color=style['color'], lw=1.15, ls=style['ls'],
                label=f'{temp} K: {final_k:.2f} ± {final_err:.2f}')
        ax.fill_between(time_ns,
                        final_k - final_err, final_k + final_err,
                        color=style['color'], alpha=0.12, lw=0)

    ax.set_xlabel('Production time (ns)', fontsize=8.5, labelpad=6)
    ax.set_ylabel(label + r' (W m$^{-1}$ K$^{-1}$)', fontsize=8.5, labelpad=6)
    ax.set_title(title_label, fontsize=9, fontweight='bold', loc='center', pad=4)
    ax.legend(loc='best', fontsize=7.2, frameon=True, framealpha=0.85,
              edgecolor='#CCCCCC')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)

fig.tight_layout(pad=0.9, w_pad=1.8)
save_figure(fig, OUTPUT, dpi=600, facecolor='white', edgecolor='none', bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print(f"Saved: {OUTPUT}")

# Convergence summary
print("\nHNEMD running-average summary:")
print(f"{'T (K)':>6} {'Dir':>6} {'κ_final':>10} {'± Err':>10} {'Err%':>7} {'1ns%':>7} {'2ns%':>7}")
print("-" * 60)
for temp in TEMPS_PLOT:
    for direction in ['para', 'perp']:
        result = load_hnemd_estimator(temp, direction)
        if result is None:
            continue
        time_ns, running_avg, final_k, final_err = result
        idx_1ns = np.searchsorted(time_ns, 1.0)
        idx_2ns = np.searchsorted(time_ns, 2.0)
        k_1ns = running_avg[min(idx_1ns, len(running_avg) - 1)]
        k_2ns = running_avg[min(idx_2ns, len(running_avg) - 1)]
        print(f"{temp:>6} {direction:>6} {final_k:>10.4f} {final_err:>10.4f} "
              f"{final_err/final_k*100:>6.1f}% {k_1ns/final_k*100:>6.1f}% {k_2ns/final_k*100:>6.1f}%")
