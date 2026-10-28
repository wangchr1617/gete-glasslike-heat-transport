#!/usr/bin/env python3
"""Fig. S2: GKMA running-integral convergence."""

from __future__ import annotations

import json
import os

from _figure_style import output_path, save_figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figure_style import (apply_style, temperature_color, TEMPERATURE_CMAP,
                           TEMPERATURE_NORM, participation_cmap, SINGLE_PANEL_SIZE)
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
GKMA = os.path.join(PROJECT_ROOT, "04_D_modal_transport", "processed")
OUTPUT = str(output_path("fig_s02_gkma_convergence.png"))
TEMPERATURES = np.asarray([300, 400, 500, 550, 585, 600, 650, 700, 800])
CMAP = TEMPERATURE_CMAP
NORM = TEMPERATURE_NORM

apply_style()

fig, ax = plt.subplots(figsize=SINGLE_PANEL_SIZE)
drifts = []
for temperature in TEMPERATURES:
    color = CMAP(NORM(temperature))
    data = np.load(os.path.join(GKMA, f"{temperature}K", "modal_results.npz"))
    metadata = json.loads(str(data["metadata_json"]))
    time = (np.arange(data["running_kappa_total_xyz_mean"].shape[1]) + 1) * metadata["sample_dt_fs"] / 1000
    running = np.mean(data["running_kappa_total_xyz_mean"], axis=0)
    running_sem = np.mean(data["running_kappa_total_xyz_sem"], axis=0)
    ax.plot(time, running, color=color, linewidth=1.3, linestyle=("-", "--", ":")[list(TEMPERATURES).index(temperature) % 3], label=f"{temperature} K")
    ax.fill_between(time, running - running_sem, running + running_sem,
                    color=color, alpha=0.08, linewidth=0)
    early = np.mean(running[(time >= 30) & (time < 40)])
    late = np.mean(running[time >= 40])
    drifts.append(100 * (late - early) / late)

ax.axvspan(30, 50, color="#999999", alpha=0.13, linewidth=0)
ax.set_xlim(0, 50)
ax.set_ylim(bottom=0)
ax.set_xlabel("Correlation time (ps)", labelpad=3)
ax.set_ylabel(r"Running $\kappa_{\rm total}$ (W m$^{-1}$ K$^{-1}$)", labelpad=3)
ax.set_ylim(0, 2.58)
ax.legend(
    loc="upper right", fontsize=6.8, ncol=3,
    frameon=True, framealpha=0.85,
    edgecolor="#CCCCCC", handlelength=1.25, columnspacing=0.7,
    borderpad=0.3, labelspacing=0.25,
)

fig.subplots_adjust(left=0.18, right=0.975, bottom=0.20, top=0.97)
save_figure(fig, OUTPUT, dpi=600, facecolor="white", edgecolor="none",
            bbox_inches=fig.bbox_inches, pad_inches=0)
plt.close(fig)
print(f"Maximum |30–40 ps vs 40–50 ps drift|: {np.max(np.abs(drifts)):.3f}%")
print(f"Saved: {OUTPUT}")
