#!/usr/bin/env python3
"""PRB Fig. 1: HNEMD conductivity tensor and anisotropy collapse."""

from __future__ import annotations

import os

from _figure_style import output_path, save_figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figure_style import (apply_style, temperature_color, TEMPERATURE_CMAP,
                           TEMPERATURE_NORM, participation_cmap)
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA = os.path.join(PROJECT_ROOT, "01_A_tensor_transport", "hnemd_tensor_source.csv")
OUTPUT = str(output_path("fig_01_transport_anisotropy.png"))
TC = 585.0
COLORS = {
    "navy": "#33539E", "sky": "#6094C3", "lavender": "#8877B1",
    "pink": "#E9B7D4", "mauve": "#A45280", "grey": "#999999",
}

sns.set_style("ticks")
sns.set_context("paper", font_scale=1.15)
apply_style()

frame = pd.read_csv(DATA).sort_values("T").reset_index(drop=True)
temperature = frame["T"].to_numpy()
trace = (2.0 * frame["kappa_perp"] + frame["kappa_para"]) / 3.0
# The source CSV uses population-normalized block errors. Table S1 and the
# figure caption report sample SEM from ten one-nanosecond blocks (ddof=1).
sample_sem_factor = np.sqrt(10.0 / 9.0)
para_sem = frame["kappa_para_err"] * sample_sem_factor
perp_sem = frame["kappa_perp_err"] * sample_sem_factor
trace_sem = np.sqrt(4.0 * perp_sem ** 2 + para_sem ** 2) / 3.0
ratio = frame["kappa_para"] / frame["kappa_perp"]
ratio_sem = ratio * np.sqrt(
    (para_sem / frame["kappa_para"]) ** 2
    + (perp_sem / frame["kappa_perp"]) ** 2
)

fig, axes = plt.subplots(
    1, 2, figsize=(6.5, 2.85),
    gridspec_kw={"width_ratios": [1, 1], "wspace": 0.27},
)
for ax in axes:
    ax.axvspan(550, 600, color=COLORS["grey"], alpha=0.15, linewidth=0, zorder=0)
    ax.axvline(TC, color=COLORS["grey"], linestyle="--", linewidth=1.0, alpha=0.7, zorder=0)
    ax.set_xlim(280, 820)

series = (
    (frame["kappa_perp"], perp_sem, "s", COLORS["navy"], r"$\kappa_\perp$"),
    (frame["kappa_para"], para_sem, "o", COLORS["sky"], r"$\kappa_\parallel$"),
    (trace, trace_sem, "D", COLORS["lavender"], r"$\kappa_{\mathrm{tr}}$"),
)
for values, errors, marker, color, label in series:
    axes[0].errorbar(
        temperature, values, yerr=errors, marker=marker, linestyle="-", color=color,
        linewidth=1.35, markersize=4.8, markerfacecolor="white", markeredgewidth=1.2,
        capsize=3, capthick=1.0, elinewidth=1.0, label=label, zorder=2,
    )
axes[0].set_ylim(0.5, 2.8)
axes[0].set_xlabel("Temperature (K)", fontsize=8.5, labelpad=6)
axes[0].set_ylabel(r"Thermal conductivity (W m$^{-1}$ K$^{-1}$)", fontsize=8.5, labelpad=6)
axes[0].set_title("(a)", fontsize=9, fontweight="bold", loc="center", pad=4)
axes[0].legend(loc="upper right", frameon=True, framealpha=0.85,
               edgecolor="#CCCCCC", fontsize=7.2, borderpad=0.35,
               handlelength=1.25, labelspacing=0.5)

ratio_handle = axes[1].errorbar(
    temperature, ratio, yerr=ratio_sem, marker="o", linestyle="-", color=COLORS["mauve"],
    linewidth=1.35, markersize=5, markerfacecolor="white", markeredgewidth=1.2,
    capsize=2, capthick=1.0, elinewidth=1.0, zorder=2,
    label=r"$\kappa_\parallel/\kappa_\perp$",
)
isotropic_handle = axes[1].axhline(1.0, color=COLORS["grey"], linestyle="--", linewidth=1.0, alpha=0.7,
                label="Isotropic limit")
axes[1].set_ylim(0.6, 1.2)
axes[1].set_xlabel("Temperature (K)", fontsize=8.5, labelpad=6)
axes[1].set_ylabel(r"Anisotropy $\kappa_\parallel/\kappa_\perp$", fontsize=8.5, labelpad=6)
axes[1].set_title("(b)", fontsize=9, fontweight="bold", loc="center", pad=4)
axes[1].legend(handles=[ratio_handle, isotropic_handle], loc="lower right",
               fontsize=7.2, frameon=True, framealpha=0.94,
               edgecolor="#CCCCCC", borderpad=0.4, handlelength=1.6,
               labelspacing=0.45)
axes[1].annotate(r"$T_c$", xy=(TC, 1.2), xytext=(4, -5),
                 textcoords="offset points", ha="left", va="top",
                 fontsize=7.2, color="#62676E")

fig.subplots_adjust(left=0.10, right=0.99, bottom=0.20, top=0.90, wspace=0.30)
save_figure(fig, OUTPUT, dpi=600, facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved: {OUTPUT}")
