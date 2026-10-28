#!/usr/bin/env python3
"""Fig. S6: effective-mode group velocities, half-linewidths, and lifetimes.

The phono3py HDF5 gamma values are half-linewidths in ordinary-frequency THz.
Lifetime in ps is 1/(4*pi*gamma), including the conversion to angular frequency.
See https://phonopy.github.io/phono3py/input-output-files.html#gamma
"""

from __future__ import annotations

import os
import json

import h5py
from _figure_style import output_path, save_figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figure_style import (apply_style, temperature_color, TEMPERATURE_CMAP,
                           TEMPERATURE_NORM, participation_cmap)
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
WTE = os.path.join(PROJECT_ROOT, "05_supplemental", "S3_effective_modes")
OUTPUT = str(output_path("fig_s06_modal_transport_descriptors.png"))
T_ALL = (300, 400, 500, 550, 585, 600, 650, 700, 800)
T_LINEWIDTH = (300, 400, 500, 600, 700, 800)

CMAP = TEMPERATURE_CMAP
NORM = TEMPERATURE_NORM

apply_style()


import phonopy
from phonopy.interface.vasp import read_vasp

group_velocity = {}
retention = {"group_velocity": {}, "linewidth": {}}
for temperature in T_ALL:
    directory = os.path.join(WTE, f"{temperature}K")
    phonon = phonopy.load(
        unitcell=read_vasp(os.path.join(directory, "POSCAR")),
        supercell_matrix=np.diag([7, 7, 7]), primitive_matrix=np.eye(3),
        force_constants_filename=os.path.join(directory, "fc2.hdf5"), produce_fc=False,
    )
    phonon.run_mesh([15, 15, 15], with_group_velocities=True)
    frequency = np.asarray(phonon.mesh.frequencies)
    speed = np.linalg.norm(np.asarray(phonon.mesh.group_velocities), axis=2)
    mask = frequency >= 0
    retention["group_velocity"][str(temperature)] = {
        "before": int(frequency.size), "after_frequency_ge_0": int(mask.sum())}
    group_velocity[temperature] = (frequency[mask], speed[mask])

linewidth = {}
for temperature in T_LINEWIDTH:
    path = os.path.join(PROJECT_ROOT, "05_supplemental", "modal_descriptors", f"{temperature}K", "kappa-m151515.hdf5")
    with h5py.File(path, "r") as stream:
        np.testing.assert_allclose(stream["temperature"][:], [temperature])
        np.testing.assert_array_equal(stream["mesh"][:], [15, 15, 15])
        frequency = np.asarray(stream["frequency"][:])
        gamma = np.asarray(stream["gamma"][0])
    mask = (frequency >= 0) & (gamma > 0)
    retention["linewidth"][str(temperature)] = {
        "before": int(frequency.size),
        "after_frequency_ge_0_gamma_gt_0": int(mask.sum())}
    linewidth[temperature] = (frequency[mask], gamma[mask])

output_path("fig_s06_retention_counts.json").write_text(
    json.dumps(retention, indent=2) + "\n", encoding="utf8")

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharex=True, layout="constrained")

for temperature in T_ALL:
    x, y = group_velocity[temperature]
    axes[0].scatter(x, y, s=3.0, color=CMAP(NORM(temperature)), alpha=0.25,
                    edgecolors="none", rasterized=True)
axes[0].set_ylabel(r"$|v_g|$ (THz Å)", labelpad=3)
axes[0].set_ylim(0, 600)
axes[0].set_title("(a)", fontsize=9, fontweight="bold", pad=3)

for temperature in T_LINEWIDTH:
    x, gamma = linewidth[temperature]
    color = CMAP(NORM(temperature))
    axes[1].scatter(x, gamma, s=3.0, color=color, alpha=0.27,
                    edgecolors="none", rasterized=True)
    axes[2].scatter(x, 1.0 / (4.0 * np.pi * gamma), s=3.0, color=color, alpha=0.27,
                    edgecolors="none", rasterized=True)
axes[1].set_yscale("log")
axes[1].set_ylim(bottom=0.1)
axes[1].set_ylabel(r"Half-linewidth $\Gamma$ (THz)", labelpad=3)
axes[1].set_title("(b)", fontsize=9, fontweight="bold", pad=3)
axes[2].set_yscale("log")
axes[2].set_ylabel(r"$\tau=1/(4\pi\Gamma)$ (ps)", labelpad=3)
axes[2].set_title("(c)", fontsize=9, fontweight="bold", pad=3)

for ax in axes:
    ax.set_xlim(0, 5.5)
    ax.set_xlabel("Frequency (THz)", labelpad=3)

scalar = matplotlib.cm.ScalarMappable(norm=NORM, cmap=CMAP)
colorbar = fig.colorbar(scalar, ax=axes, pad=0.015, aspect=24, shrink=0.90)
colorbar.set_label("Temperature (K)", fontsize=8, labelpad=3)
colorbar.set_ticks([300, 400, 500, 585, 700, 800])
colorbar.ax.tick_params(labelsize=7.0, direction="in")

save_figure(fig, OUTPUT, dpi=600, facecolor="white", edgecolor="none",
            bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print(f"Saved: {OUTPUT}")
