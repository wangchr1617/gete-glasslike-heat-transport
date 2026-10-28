#!/usr/bin/env python3
"""Main Fig. 2: PJT dynamics with six independent 585 K trajectories."""

from __future__ import annotations

import csv
import json
import os

from _figure_style import output_path, save_figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figure_style import (apply_style, temperature_color, TEMPERATURE_CMAP,
                           TEMPERATURE_NORM, participation_cmap)
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import seaborn as sns


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA = os.path.join(PROJECT_ROOT, "02_B_pjt_dynamics", "replicas")
OUTPUT_STEM = str(output_path("fig_02_pjt_dynamics"))
OUTPUT = f"{OUTPUT_STEM}.png"
TEMPERATURES = (550, 585, 650)
REPLICAS = {550: (1, 2, 3), 585: (1, 2, 3, 4, 5, 6), 650: (1, 2, 3)}
COLORS = {
    "navy": "#33539E", "sky": "#6094C3", "lavender": "#8877B1",
    "pink": "#E9B7D4", "mauve": "#A45280", "grey": "#999999",
}
TEMP_COLORS = {t: temperature_color(t) for t in TEMPERATURES}
TEMP_MARKERS = {550: "o", 585: "s", 650: "D"}

sns.set_style("ticks")
sns.set_context("paper", font_scale=1.15)
apply_style()


def read_csv(path: str) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty data file: {path}")
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}


def sem(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.std(values, axis=axis, ddof=1) / np.sqrt(values.shape[axis])


summaries, correlations, spectra = {}, {}, {}
reference_time = reference_frequency = None
for temperature in TEMPERATURES:
    summaries[temperature], corr_reps, spec_reps = [], [], []
    for replica in REPLICAS[temperature]:
        analysis = os.path.join(DATA, f"{temperature}K", f"traj_{replica:02d}")
        with open(os.path.join(analysis, "summary.json"), encoding="utf-8") as stream:
            summary = json.load(stream)
        if summary["temperature_K"] != temperature or summary["replica"] != replica:
            raise ValueError(f"Metadata mismatch: {analysis}")
        summaries[temperature].append(summary)
        correlation = read_csv(os.path.join(analysis, "correlation.csv"))
        spectrum = read_csv(os.path.join(analysis, "spectrum.csv"))
        if reference_time is None:
            reference_time = correlation["time_ps"]
            reference_frequency = spectrum["frequency_THz"]
        if not np.array_equal(correlation["time_ps"], reference_time):
            raise ValueError("Correlation time grids differ")
        if not np.array_equal(spectrum["frequency_THz"], reference_frequency):
            raise ValueError("Spectrum frequency grids differ")
        corr_reps.append(correlation)
        spec_reps.append(spectrum)
    correlations[temperature] = {key: np.stack([item[key] for item in corr_reps]) for key in corr_reps[0]}
    spectra[temperature] = {key: np.stack([item[key] for item in spec_reps]) for key in spec_reps[0]}


def aggregate(key: str) -> tuple[np.ndarray, np.ndarray]:
    values = [np.asarray([entry[key] for entry in summaries[t]], dtype=float) for t in TEMPERATURES]
    return np.asarray([np.mean(v) for v in values]), np.asarray([sem(v) for v in values])


local, local_sem = aggregate("mean_local_pjt_A")
collective, collective_sem = aggregate("mean_collective_order_ratio")
tau_local, tau_local_sem = aggregate("connected_relaxation_time_ps")
tau_collective, tau_collective_sem = aggregate("collective_relaxation_time_ps")
low, low_sem = aggregate("low_frequency_weight_0_0p5_THz")

fig = plt.figure(figsize=(7.2, 3.05))
grid = fig.add_gridspec(
    1, 3, width_ratios=(1.16, 1.48, 1.16), wspace=0.2,
    left=0.070, right=0.992, bottom=0.21, top=0.90,
)

# (a) Discrete state portrait. No line is drawn between the three temperatures.
ax = fig.add_subplot(grid[0, 0])
marker_area = np.full(3, 42.0)  # Equal area; shape identifies temperature.
for index, temperature in enumerate(TEMPERATURES):
    ax.errorbar(
        local[index], collective[index], xerr=local_sem[index], yerr=collective_sem[index],
        marker="none", color=TEMP_COLORS[temperature], linestyle="none",
        capsize=2.0, capthick=0.75, elinewidth=0.75, zorder=3,
    )
    ax.scatter(
        local[index], collective[index], s=marker_area[index],
        marker=TEMP_MARKERS[temperature], facecolor="white",
        edgecolor=TEMP_COLORS[temperature], linewidth=1.3,
        zorder=5,
    )
ax.set_xlim(0.338, 0.364)
ax.set_ylim(0, 0.53)
ax.set_xlabel("Mean local PJT amplitude (Å)", labelpad=3)
ax.set_ylabel("Collective/local PJT ratio", labelpad=3)
ax.set_title("(a)", fontsize=9, fontweight="bold", loc="center", pad=3)
label_positions = {
    550: (0.350, 0.500),
    585: (0.344, 0.105),
    650: (0.355, 0.042),
}
for temperature in TEMPERATURES:
    ax.text(
        *label_positions[temperature], f"{temperature} K",
        color=TEMP_COLORS[temperature], fontsize=6.9,
        ha="left", va="center",
    )

# A light outline separates the numerical inset from the state-portrait axes.
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((0.06, 0.41), 0.92, 0.425, transform=ax.transAxes,
                       fill=False, edgecolor="#B0B0B0", linewidth=0.45, zorder=1))
# Direct numerical encoding makes the 585/550 K relaxation ratio auditable.
ax.text(0.09, 0.78, "Collective relaxation (ps)", transform=ax.transAxes,
        fontsize=6.9, weight="bold", va="center")
for i, temperature in enumerate(TEMPERATURES):
    y = 0.68 - 0.11*i
    ax.text(0.09, y, f"{temperature} K", transform=ax.transAxes, fontsize=6.9,
            color=TEMP_COLORS[temperature], va="center")
    ax.text(0.95, y, f"{tau_collective[i]:.3f} ± {tau_collective_sem[i]:.3f}",
            transform=ax.transAxes, fontsize=6.9, ha="right", va="center")

# (b) Replica-resolved local and collective connected correlations.
# Fixed inset geometry avoids the excessive whitespace that nested constrained
# layouts can create for heatmaps with different physical time windows.
panel_b = fig.add_subplot(grid[0, 1])
panel_b.set_axis_off()
panel_b.set_title("(b)", fontsize=9, fontweight="bold", pad=3)
ax_local = panel_b.inset_axes([0.07, 0.63, 0.79, 0.29])
ax_collective = panel_b.inset_axes([0.07, 0.08, 0.79, 0.29])
cax = panel_b.inset_axes([0.905, 0.08, 0.026, 0.84])
local_rows = np.asarray([row for t in TEMPERATURES for row in correlations[t]["C_self_connected"]])
collective_rows = np.asarray([row for t in TEMPERATURES for row in correlations[t]["C_collective_connected"]])
assert local_rows.shape[0] == collective_rows.shape[0] == 12
time = reference_time
local_use, collective_use = time <= 0.5, time <= 10.0
norm = TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=1.0)
image = ax_local.imshow(
    local_rows[:, local_use], aspect="auto", origin="upper",
    extent=(time[local_use][0], time[local_use][-1], 11.5, -0.5),
    cmap="PuOr_r", norm=norm, interpolation="nearest", rasterized=True,
)
ax_collective.imshow(
    collective_rows[:, collective_use], aspect="auto", origin="upper",
    extent=(time[collective_use][0], time[collective_use][-1], 11.5, -0.5),
    cmap="PuOr_r", norm=norm, interpolation="nearest", rasterized=True,
)
for heat_axis in (ax_local, ax_collective):
    heat_axis.axhline(2.5, color="white", linewidth=1.0)
    heat_axis.axhline(8.5, color="white", linewidth=1.0)
    heat_axis.set_ylim(11.5, -0.5)
ax_local.set_yticks((1, 5.5, 10), labels=("550", "585", "650"))
ax_collective.set_yticks((1, 5.5, 10), labels=("550", "585", "650"))
ax_local.set_xlabel("Lag time (ps)", labelpad=3)
ax_collective.set_xlabel("Lag time (ps)", labelpad=3)
ax_local.set_ylabel("T (K)", labelpad=2)
ax_collective.set_ylabel("T (K)", labelpad=2)
ax_local.set_title("Local", fontsize=8.0, fontweight="bold", loc="center", pad=2)
ax_collective.set_title("Collective", fontsize=8.2, fontweight="bold", loc="center", pad=2)
colorbar = fig.colorbar(image, cax=cax)
colorbar.ax.set_title(r"$C_{\rm conn}$", fontsize=8.2, pad=5)
colorbar.ax.tick_params(direction="in", length=3.0, width=0.6, labelsize=6.8)

# Align the central heatmaps with the adjacent panels.
pos = panel_b.get_position()
panel_b.set_position([pos.x0 - 0.01, pos.y0, pos.width, pos.height])

# (c) A common quantitative ordinate allows direct comparison of the spectra.
ax = fig.add_subplot(grid[0, 2])
frequency = reference_frequency
use = (frequency >= 0) & (frequency <= 5.0)
frequency = frequency[use]
means = {t: np.mean(spectra[t]["normalized_density"][:, use], axis=0) for t in TEMPERATURES}
errors = {t: sem(spectra[t]["normalized_density"][:, use], axis=0) for t in TEMPERATURES}
for temperature, linestyle in zip(TEMPERATURES, ("-", "--", ":")):
    mean, error = means[temperature], errors[temperature]
    color = TEMP_COLORS[temperature]
    ax.fill_between(
        frequency, np.maximum(mean - error, 0), mean + error,
        color=color, alpha=0.13, linewidth=0,
    )
    ax.plot(frequency, mean, color=color, linestyle=linestyle,
            linewidth=1.35, label=f"{temperature} K")
ax.set_xlim(0, 5)
ax.set_ylim(0, 0.65)
ax.set_yticks((0, 0.2, 0.4, 0.6))
ax.set_xlabel("Frequency (THz)", labelpad=3)
ax.set_ylabel(r"Normalized spectral density (THz$^{-1}$)", labelpad=1)
ax.set_title("(c)", fontsize=9, fontweight="bold", loc="center", pad=3)
ax.legend(loc="upper right", fontsize=6.9, frameon=False)

save_figure(fig, OUTPUT, dpi=600, facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=0.05)

plt.close(fig)

for index, temperature in enumerate(TEMPERATURES):
    print(
        f"{temperature} K: local={local[index]:.5f}+/-{local_sem[index]:.5f} A; "
        f"collective/local={collective[index]:.4f}+/-{collective_sem[index]:.4f}; "
        f"tau_local={tau_local[index]:.4f}+/-{tau_local_sem[index]:.4f} ps; "
        f"tau_collective={tau_collective[index]:.3f}+/-{tau_collective_sem[index]:.3f} ps; "
        f"W_0-0.5={low[index]:.4f}+/-{low_sem[index]:.4f}"
    )
print(f"Saved: {OUTPUT}")
