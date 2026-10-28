#!/usr/bin/env python3
"""
Post-process GPUMD force.out → HiPhive-compatible extended XYZ.

Usage:
    python extract_forces.py <workdir>

Reads <workdir>/displacements/structure_XXXX.xyz and
<workdir>/force_eval/structure_XXXX/force.out,
writes <workdir>/displaced_structures_with_forces.extxyz.
"""

import sys, os, re
import numpy as np

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <workdir>")
    sys.exit(1)

WORKDIR = os.path.abspath(sys.argv[1])
DISPLACEMENTS_DIR = os.path.join(WORKDIR, 'displacements')
FORCE_EVAL_DIR = os.path.join(WORKDIR, 'force_eval')
OUTPUT_FILE = os.path.join(WORKDIR, 'displaced_structures_with_forces.extxyz')


def parse_force_out(filepath, n_atoms):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"force.out not found: {filepath}")
    with open(filepath) as f:
        lines = [l for l in f if not l.startswith('#') and l.strip()]
    if len(lines) < n_atoms:
        raise ValueError(f"Expected >= {n_atoms} lines, got {len(lines)}")
    forces = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        parts = lines[i].split()
        forces[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
    return forces


def read_extxyz(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    m = re.search(r'Lattice="([^"]+)"', comment)
    if not m:
        raise ValueError(f"No Lattice in: {comment}")
    cell = np.array([float(x) for x in m.group(1).split()]).reshape(3, 3)
    species, pos = [], np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        parts = lines[2 + i].split()
        species.append(parts[0])
        pos[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return cell, species, pos


def main():
    structure_files = sorted([
        f for f in os.listdir(DISPLACEMENTS_DIR)
        if f.startswith('structure_') and f.endswith('.xyz')
    ])
    if len(structure_files) != 500:
        raise ValueError(f"Expected 500 configurations, found {len(structure_files)}")
    if not structure_files:
        print("ERROR: No displacement files found.")
        sys.exit(1)

    _, species, _ = read_extxyz(os.path.join(DISPLACEMENTS_DIR, structure_files[0]))
    n_atoms = len(species)
    n_ge, n_te = species.count('Ge'), species.count('Te')
    print(f"Processing {len(structure_files)} structures "
          f"({n_atoms} atoms: {n_ge} Ge + {n_te} Te)...")

    for fname in structure_files:
        idx = int(re.match(r"structure_(\d+)\.xyz", fname).group(1))
        parse_force_out(os.path.join(FORCE_EVAL_DIR, f"structure_{idx:04d}", "force.out"), n_atoms)
    if os.path.exists(OUTPUT_FILE):
        raise FileExistsError(OUTPUT_FILE)
    n_success = n_missing = n_parse_err = 0
    with open(OUTPUT_FILE, 'w') as out_f:
        for i, fname in enumerate(structure_files):
            idx = int(re.match(r'structure_(\d+)\.xyz', fname).group(1))
            disp_file = os.path.join(DISPLACEMENTS_DIR, fname)
            force_file = os.path.join(FORCE_EVAL_DIR,
                                      f'structure_{idx:04d}', 'force.out')
            try:
                cell_i, species_i, pos_i = read_extxyz(disp_file)
                forces_i = parse_force_out(force_file, n_atoms)
                lattice_str = ' '.join(f'{v:.10f}' for v in cell_i.flatten())
                out_f.write(f'{n_atoms}\n')
                out_f.write(f'Lattice="{lattice_str}" '
                            f'Properties=species:S:1:pos:R:3:forces:R:3 '
                            f'pbc="T T T"\n')
                for j in range(n_atoms):
                    out_f.write(f'{species_i[j]} '
                                f'{pos_i[j, 0]:.10f} {pos_i[j, 1]:.10f} '
                                f'{pos_i[j, 2]:.10f} '
                                f'{forces_i[j, 0]:.10f} {forces_i[j, 1]:.10f} '
                                f'{forces_i[j, 2]:.10f}\n')
                n_success += 1
            except FileNotFoundError:
                n_missing += 1
                if n_missing <= 5:
                    print(f"  MISSING: structure_{idx:04d}/force.out")
            except Exception as e:
                n_parse_err += 1
                if n_parse_err <= 5:
                    print(f"  ERROR structure_{idx:04d}: {e}")
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(structure_files)}]")

    print(f"\nResults: {n_success} success, {n_missing} missing, "
          f"{n_parse_err} errors")
    print(f"Output: {OUTPUT_FILE}")
    if n_missing or n_parse_err:
        raise RuntimeError("Incomplete force dataset; do not use this file for fitting")
    print("Next: python minimal_reproduction/force_constants/extract_ifcs.py <workdir>")


if __name__ == '__main__':
    main()
