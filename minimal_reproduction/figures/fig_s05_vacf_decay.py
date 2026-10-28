#!/usr/bin/env python3
"""
fig_s05_vacf_decay.py
2D heatmap of the logarithm of the normalized VACF envelope, log10|C(t,T)|,
versus correlation time and temperature. All stored values are finite; nearest
cell rendering is used so the non-uniform temperature grid is not visually
interpolated into apparent blank or mismatched pixels.
"""
from _figure_style import output_path, save_figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _figure_style import (apply_style, temperature_color, TEMPERATURE_CMAP,
                           TEMPERATURE_NORM, participation_cmap, SINGLE_PANEL_SIZE)
import seaborn as sns
import numpy as np
import os
from scipy.signal import hilbert

# === Paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VACF_DIR = os.path.join(PROJECT_ROOT, '05_supplemental', 'S2_vacf')
OUTPUT = str(output_path('fig_s05_vacf_decay.png'))

TC = 585.0

# === Temperatures (17 points, same as Grüneisen data) ===
TEMPS = [300, 400, 500, 550, 560, 570, 580, 585, 590,
         600, 610, 620, 630, 640, 650, 700, 800]
N_T = len(TEMPS)

# === Color palette ===
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
sns.set_context("paper", font_scale=1.1)
apply_style()

# === Load MVAC for all temperatures ===
t_common = None
envelope_grid = []

for T in TEMPS:
    path = os.path.join(VACF_DIR, f'{T}K', 'mvac.out')
    raw = np.loadtxt(path)
    t = raw[:, 0]
    vacf_avg = (raw[:, 1] + raw[:, 2] + raw[:, 3]) / 3.0
    if t_common is None:
        t_common = t
    analytic = hilbert(vacf_avg)
    envelope = np.abs(analytic)
    min_val = 1e-8
    # The Hilbert transform has a small edge overshoot above the normalized
    # t=0 value. Clip only this numerical artefact to retain the [0, 1] scale.
    envelope = np.clip(envelope, min_val, 1.0)
    log_envelope = np.log10(envelope)
    envelope_grid.append(log_envelope)

envelope_grid = np.array(envelope_grid)  # (N_T, N_time)
if not np.isfinite(envelope_grid).all():
    raise ValueError("VACF envelope contains non-finite values")

# === Restrict to t <= 8 ps ===
time_mask = t_common <= 8.0
t_plot = t_common[time_mask]
decay_plot = envelope_grid[:, time_mask]

# === Plot ===
fig = plt.figure(figsize=SINGLE_PANEL_SIZE)
ax = fig.add_axes([0.17, 0.20, 0.66, 0.74])

im = ax.pcolormesh(t_plot, np.array(TEMPS), decay_plot,
                   cmap='viridis', vmin=-5.5, vmax=0.0,
                   shading='nearest', rasterized=True)

cax = fig.add_axes([0.855, 0.24, 0.025, 0.66])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r'$\log_{10}|C(t,T)|$', fontsize=8)
cb.set_ticks([-5, -4, -3, -2, -1, 0])
cb.ax.tick_params(labelsize=7.2)

# Contour lines at selected log levels
log_levels = [-3.0, -2.0, -1.0]
ax.contour(t_plot, np.array(TEMPS), decay_plot,
           levels=log_levels, colors='#E5E9ED', linewidths=0.20, alpha=0.20)

# Tc reference line
ax.axhline(y=TC, color='#30343B', ls='--', lw=0.8, alpha=0.85)

ax.set_xlabel('Correlation time t (ps)', fontsize=8.5, labelpad=4)
ax.set_ylabel('Temperature (K)', fontsize=8.5, labelpad=4)
ax.set_xlim(t_plot[0], t_plot[-1])
ax.set_ylim(300, 800)
ax.tick_params(axis='both', labelsize=7.2)

save_figure(fig, OUTPUT, dpi=600, facecolor='white', edgecolor='none',
            bbox_inches=fig.bbox_inches, pad_inches=0)
plt.close(fig)
print(f"Saved: {OUTPUT}")
