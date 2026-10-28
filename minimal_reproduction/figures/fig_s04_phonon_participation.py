#!/usr/bin/env python3
"""
fig_s04_phonon_participation.py
Phonon band structure coloured by participation ratio PPR — 9 temperatures.
PPR = 1: equal Ge/Te species weights; PPR = 0.5: one-species weight.
This two-species measure does not classify propagation or spatial localization.

Data: HiPhive CS FORCE_CONSTANTS (primitive cell, 7³ supercell).
k-path: Standard R3m primitive-cell path, densified from seekpath anchors.

Requires: conda activate thc (phonopy + ase + seekpath).
"""

from _figure_style import output_path, save_figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _figure_style import (apply_style, temperature_color, TEMPERATURE_CMAP,
                           TEMPERATURE_NORM, participation_cmap)
import seaborn as sns
import numpy as np
import seekpath
import os

# === Paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
WTE_RTA = os.path.join(PROJECT_ROOT, '05_supplemental', 'S3_effective_modes')
OUTPUT = str(output_path('fig_s04_phonon_participation.png'))

# === Constants ===
TEMPS = [300, 400, 500, 550, 585, 600, 650, 700, 800]
FREQ_MAX = 5.5
PPR_VMIN = 0.50
PPR_VMAX = 1.00
DENSIFY = 4        # insert DENSIFY-1 extra points between each seekpath anchor pair

# === Styling ===
GREY = '#78838F'
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.15)
apply_style()


def build_dense_path(atoms):
    """
    Use seekpath to get the standard k-path, then densify.
    Jumps between disconnected segments (H_2→H_0, S_0→S_2) keep
    zero linear-coordinate change: both endpoints share the same x-pos.
    """
    scaled_positions = (
        atoms.get_scaled_positions()
        if hasattr(atoms, 'get_scaled_positions')
        else atoms.scaled_positions
    )
    struct = (atoms.cell, scaled_positions, atoms.numbers)
    sp = seekpath.get_explicit_k_path(struct)

    kp_rel = np.array(sp['explicit_kpoints_rel'])        # (N, 3)
    kp_lin = np.array(sp['explicit_kpoints_linearcoord']) # (N,)
    labels = sp['explicit_kpoints_labels']

    all_q = []
    all_dist = []

    for i in range(len(kp_rel) - 1):
        start_q = kp_rel[i]
        end_q = kp_rel[i + 1]
        start_d = kp_lin[i]
        end_d = kp_lin[i + 1]

        for j in range(1 if abs(end_d-start_d) < 1e-12 else DENSIFY):
            frac = j / DENSIFY  # No artificial interpolation across path discontinuities.
            all_q.append(start_q + frac * (end_q - start_q))
            all_dist.append(start_d + frac * (end_d - start_d))

    # Don't forget the very last point
    all_q.append(kp_rel[-1])
    all_dist.append(kp_lin[-1])

    # Build tick list: keep labelled points, deduplicate neighbours
    tick_pos, tick_lbl = [], []
    for i, lab in enumerate(labels):
        if not lab:
            continue
        pos = kp_lin[i]
        lbl = lab.replace('GAMMA', r'$\Gamma$')
        if '_' in lbl:
            lbl = lbl.replace('_', r'$_') + r'$'
        if tick_pos and abs(pos - tick_pos[-1]) < 0.001:
            if lbl != tick_lbl[-1]:
                tick_lbl[-1] += '\n' + lbl
            continue
        tick_pos.append(pos)
        tick_lbl.append(lbl)

    return np.array(all_q), np.array(all_dist), tick_pos, tick_lbl


def compute_ppr_from_eigvecs(eigvecs, n_atom=2):
    """PPR from eigenvector array of shape (..., n_dof)."""
    ppr = np.zeros(eigvecs.shape[:-1])
    for idx in np.ndindex(eigvecs.shape[:-1]):
        e = eigvecs[idx]
        e_atom = np.zeros(n_atom)
        for ia in range(n_atom):
            e_atom[ia] = np.sum(np.abs(e[3 * ia:3 * ia + 3]) ** 2)
        ppr[idx] = 1.0 / (n_atom * np.sum(e_atom ** 2))
    return ppr


# =====================================================================
# Main
# =====================================================================

import phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.vasp import read_vasp

# Build dense k-path
prim0 = read_vasp(os.path.join(WTE_RTA, '300K', 'POSCAR'))
all_q, all_dist, tick_pos, tick_lbl = build_dense_path(prim0)
seg_str = ' → '.join(tick_lbl)
print(f"Band path: {len(all_q)} q-points")
print(f"  Labels: {seg_str}")

# Collect band + PPR data
results = {}
for T in TEMPS:
    wdir = os.path.join(WTE_RTA, f'{T}K')
    print(f"  {T} K …", end=' ', flush=True)

    cell = read_vasp(os.path.join(wdir, 'POSCAR'))
    ph = phonopy.load(
        unitcell=cell,
        supercell_matrix=np.diag([7, 7, 7]),
        primitive_matrix=np.eye(3),
        force_constants_filename=os.path.join(wdir, 'fc2.hdf5'),
        is_symmetry=True,
        produce_fc=False,
    )
    ph.run_band_structure([all_q], with_eigenvectors=True)

    freqs = ph.band_structure.frequencies[0]
    eigvecs = ph.band_structure.eigenvectors[0]
    # Phonopy columns are modes; the PPR helper expects one mode per row.
    # https://phonopy.github.io/phonopy/phonopy-module.html#eigenvectors
    mode_vectors = np.swapaxes(eigvecs, -1, -2)
    ppr = compute_ppr_from_eigvecs(mode_vectors, n_atom=2)
    assert np.allclose(np.sum(np.abs(mode_vectors)**2, axis=-1), 1.0)
    assert np.min(ppr) >= 0.5-1e-10 and np.max(ppr) <= 1.0+1e-10
    # Algebraic eigenpair check at one generic q point.
    ph.dynamical_matrix.run(all_q[7])
    dm = ph.dynamical_matrix.dynamical_matrix
    eigenvalues = np.linalg.eigvalsh(dm)
    residual = np.linalg.norm(dm @ eigvecs[7] - eigvecs[7] * eigenvalues)
    assert residual < 1e-8, residual

    n_imag = (freqs < -0.01).sum()
    print(f"({n_imag} imag modes)")

    results[T] = {'freqs': freqs, 'ppr': ppr}

# ---- Plot: 3 × 3 panels ----
fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.65), sharey=True, sharex=True,
                         constrained_layout=True)

cmap = participation_cmap()

for iT, T in enumerate(TEMPS):
    ax = axes.flat[iT]
    ax.set_box_aspect(1)
    freqs = results[T]['freqs']
    ppr = results[T]['ppr']
    n_bands = freqs.shape[1]

    for ib in range(n_bands):
        f = freqs[:, ib]
        mask = np.ones(f.shape, dtype=bool)  # Retain the complete displayed dispersion.
        if mask.sum() == 0:
            continue
        sc = ax.scatter(all_dist[mask], f[mask], c=ppr[mask, ib],
                        cmap=cmap, vmin=PPR_VMIN, vmax=PPR_VMAX,
                        s=2.2, edgecolors='none', linewidths=0, rasterized=True)

    # Vertical lines at high-symmetry points
    for xp in tick_pos:
        ax.axvline(x=xp, color=GREY, lw=0.45, alpha=0.35)

    # Zero-frequency line
    ax.axhline(y=0.0, color=GREY, lw=0.45, alpha=0.35)

    # Ticks
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, fontsize=7.2)
    ax.set_ylim(-2.05, FREQ_MAX)
    ax.tick_params(axis="y", labelsize=7.2)
    ax.set_xlim(all_dist[0], all_dist[-1])

    ax.set_title(f'{T} K', fontsize=8, fontweight='bold', loc='center', pad=4)
    ax.set_ylabel('Frequency (THz)', fontsize=8, labelpad=4)

# Hide redundant labels
for ax in axes[:, 1:].flat:
    ax.set_ylabel('')
for ax in axes[:-1, :].flat:
    ax.set_xlabel('')
for ax in axes[-1, :].flat:
    ax.set_xlabel('Wave vector', fontsize=8, labelpad=4)

# Shared colour bar
cbar = fig.colorbar(sc, ax=axes, orientation='vertical', shrink=0.85, pad=0.02,
                    label='Two-species participation ratio', fraction=0.035)
cbar.ax.tick_params(labelsize=7.2)
cbar.ax.yaxis.label.set_size(8)

save_figure(fig, OUTPUT, facecolor='white', edgecolor='none',
            bbox_inches='tight', pad_inches=0.05, dpi=600)
fig.canvas.draw()
for ax in axes.flat:
    extent=ax.get_window_extent()
    assert abs(extent.width/extent.height-1) < 0.001
plt.close(fig)
print(f"\nSaved: {OUTPUT}")
