#!/usr/bin/env python3
"""
Extract averaged primitive cell from NEP-relaxed 7x7x7 supercell.

Usage:
    python extract_primitive.py <workdir>

Reads <workdir>/model.xyz (relaxed supercell) and writes
<workdir>/primitive.xyz (averaged 2-atom primitive cell).

Uses position-based unfolding: each atom's cell index is determined
by rounding f_sc * N_REPEAT, NOT by its index in the file. This is
robust against thermal drift and PBC wrapping during NVT relaxation.
"""

import sys, os
import numpy as np
from ase.io import read, write
from ase import Atoms

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <workdir>")
    sys.exit(1)

WORKDIR = os.path.abspath(sys.argv[1])
SUPERCELL_FILE = os.path.join(WORKDIR, 'model.xyz')
OUTPUT_FILE = os.path.join(WORKDIR, 'primitive.xyz')
N_REPEAT = (7, 7, 7)

if not os.path.exists(SUPERCELL_FILE):
    print(f"ERROR: {SUPERCELL_FILE} not found. Run relaxation first.")
    sys.exit(1)

sc = read(SUPERCELL_FILE)
n_total = len(sc)
n_prim = n_total // (N_REPEAT[0] * N_REPEAT[1] * N_REPEAT[2])

print(f"Supercell: {n_total} atoms")
print(f"Primitive: {n_prim} atoms")
print(f"Repeat: {N_REPEAT[0]}x{N_REPEAT[1]}x{N_REPEAT[2]} = "
      f"{N_REPEAT[0]*N_REPEAT[1]*N_REPEAT[2]} cells")

# Cell: supercell / 7 -> primitive
prim_cell = sc.cell.array.copy()
prim_cell[0] /= N_REPEAT[0]
prim_cell[1] /= N_REPEAT[1]
prim_cell[2] /= N_REPEAT[2]

# Position-based unfolding: for each atom, find which cell it's in
# by rounding its unfolded coordinate to the nearest integer.
# Then subtract the cell index to get the primitive-cell fractional coordinate.
sc_pos = sc.get_scaled_positions()
symbols = sc.get_chemical_symbols()

# Group atoms by species
species_list = list(dict.fromkeys(symbols))  # preserve order
prim_frac = []

for species in species_list:
    mask = np.array([s == species for s in symbols])
    pos = sc_pos[mask]  # (343, 3)

    unfolded_raw = pos * np.array(N_REPEAT)  # (343, 3)
    cell_idx = np.round(unfolded_raw).astype(int)  # nearest integer cell
    f_prim_all = unfolded_raw - cell_idx  # primitive frac coords

    # Wrap to [-0.5, 0.5) for averaging (handles PBC)
    f_prim_all = (f_prim_all + 0.5) % 1.0 - 0.5

    avg_frac = f_prim_all.mean(axis=0)
    # Shift to [0, 1) for output
    avg_frac_out = avg_frac % 1.0
    # Handle floating-point noise near 0 or 1
    avg_frac_out = np.where(np.abs(avg_frac_out) < 1e-6, 0.0, avg_frac_out)
    avg_frac_out = np.where(np.abs(avg_frac_out - 1.0) < 1e-6, 0.0, avg_frac_out)
    std_frac = f_prim_all.std(axis=0)

    prim_frac.append(avg_frac_out)

    print(f"  {species}: frac=({avg_frac_out[0]:.8f}, {avg_frac_out[1]:.8f}, "
          f"{avg_frac_out[2]:.8f}) std=({std_frac[0]:.6f},{std_frac[1]:.6f},{std_frac[2]:.6f})")

    # Sanity: check that all unfolded positions are nearly identical
    max_spread = np.max(np.abs(f_prim_all - f_prim_all.mean(axis=0)))
    print(f"    Max spread: {max_spread:.6f} (should be small, <~0.01)")

prim = Atoms(symbols=species_list,
             scaled_positions=np.array(prim_frac),
             cell=prim_cell, pbc=True)
write(OUTPUT_FILE, prim, format='extxyz')
print(f"\nSaved: {OUTPUT_FILE}")
print(f"  Volume: {prim.get_volume():.4f} A^3")
print(f"  Angles: {prim.cell.cellpar()[3:].round(2)} deg")
