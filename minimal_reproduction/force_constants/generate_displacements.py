#!/usr/bin/env python3
"""
Generate HiPhive displacement configurations from relaxed supercell.

Usage:
    python generate_displacements.py <workdir>

Reads <workdir>/model.xyz (relaxed supercell), writes:
  - <workdir>/displaced_structures.extxyz  (combined, for HiPhive)
  - <workdir>/displacements/structure_XXXX.xyz  (individual, for GPUMD)
"""

import sys, os
import numpy as np
from ase.io import read, write

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <workdir>")
    sys.exit(1)

WORKDIR = os.path.abspath(sys.argv[1])
SUPERCELL_FILE = os.path.join(WORKDIR, 'model.xyz')
OUTPUT_EXTXYZ = os.path.join(WORKDIR, 'displaced_structures.extxyz')
DISPLACEMENTS_DIR = os.path.join(WORKDIR, 'displacements')

N_STRUCTURES = 500
RATTLE_STD = 0.03     # A
CUTOFF = 3.0          # 3*sigma
SEED = 42

if not os.path.exists(SUPERCELL_FILE):
    print(f"ERROR: {SUPERCELL_FILE} not found. Run relaxation first.")
    sys.exit(1)

if os.path.exists(OUTPUT_EXTXYZ) or os.path.exists(DISPLACEMENTS_DIR):
    raise FileExistsError("Displacement outputs already exist")

atoms = read(SUPERCELL_FILE)
n_atoms = len(atoms)
print(f"Supercell: {n_atoms} atoms ({atoms.get_chemical_symbols()[:2]}...)")

np.random.seed(SEED)
max_disp = CUTOFF * RATTLE_STD
structures = []
n_resampled = 0

for i in range(N_STRUCTURES):
    rattled = atoms.copy()
    disps = np.random.normal(0, RATTLE_STD, (n_atoms, 3))
    for _ in range(100):
        too_large = np.abs(disps) > max_disp
        if too_large.any():
            n_bad = too_large.sum()
            n_resampled += n_bad
            disps[too_large] = np.random.normal(0, RATTLE_STD, n_bad)
        else:
            break
    rattled.positions += disps
    rattled.wrap()
    structures.append(rattled)

if n_resampled > 0:
    print(f"  Resampled {n_resampled}/{N_STRUCTURES*n_atoms*3} "
          f"displacements (>{max_disp:.2f} A)")

# Combined file
write(OUTPUT_EXTXYZ, structures, format='extxyz')
print(f"  -> {OUTPUT_EXTXYZ} ({len(structures)} frames)")

# Individual files
os.makedirs(DISPLACEMENTS_DIR, exist_ok=True)
for i, at in enumerate(structures):
    fname = os.path.join(DISPLACEMENTS_DIR, f'structure_{i:04d}.xyz')
    write(fname, at, format='extxyz')
print(f"  -> {DISPLACEMENTS_DIR}/structure_0000.xyz ~ "
      f"structure_{len(structures)-1:04d}.xyz")

# Statistics
cell = atoms.cell.array
ref_scaled = atoms.get_scaled_positions()
all_disps = []
for at in structures[:50]:  # sample 50 for stats
    diff_scaled = at.get_scaled_positions() - ref_scaled
    diff_scaled -= np.round(diff_scaled)
    diff_cart = diff_scaled @ cell
    all_disps.append(np.linalg.norm(diff_cart, axis=1))
all_disps = np.concatenate(all_disps)
print(f"  Mean |dr| = {np.mean(all_disps):.5f} A "
      f"(untruncated Gaussian expectation {2*RATTLE_STD*np.sqrt(2/np.pi):.4f})")
print("Generated the displacement dataset.")
