#!/usr/bin/env python3
"""Extract time-domain local PJT-vector observables from a GPUMD dump.xyz.

The PJT vector of a Ge site is its displacement from the instantaneous centre
of the same six nearest Te neighbours identified in model.xyz.  It computes local-coordinate spectra and connected local/collective correlations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


DT_PS = 0.0005 * 40  # 0.5 fs timestep, dump every 40 steps
EXPECTED_FRAMES = 200000 // 40


def parse_lattice(header: str) -> np.ndarray:
    match = re.search(r'Lattice="([^"]+)"', header)
    if not match:
        raise ValueError("No Lattice field in extended-XYZ header")
    values = np.fromstring(match.group(1), sep=" ")
    if values.size != 9:
        raise ValueError("Lattice field must contain nine numbers")
    return values.reshape(3, 3)


def read_xyz_frame(stream, expected_atoms: int | None = None):
    first = stream.readline()
    if not first:
        return None
    natoms = int(first.strip())
    if expected_atoms is not None and natoms != expected_atoms:
        raise ValueError(f"Frame has {natoms} atoms; expected {expected_atoms}")
    header = stream.readline().strip()
    species = np.empty(natoms, dtype="U2")
    positions = np.empty((natoms, 3), dtype=np.float64)
    for index in range(natoms):
        fields = stream.readline().split()
        if len(fields) < 4:
            raise ValueError(f"Malformed atom line {index + 1}")
        species[index] = fields[0]
        positions[index] = fields[1:4]
    return species, positions, header


def read_model(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        frame = read_xyz_frame(stream)
    if frame is None:
        raise ValueError(f"Empty model: {path}")
    species, positions, header = frame
    return species, positions, parse_lattice(header)


def orthogonal_basis(lattice: np.ndarray):
    lengths = np.linalg.norm(lattice, axis=1)
    basis = lattice / lengths[:, None]
    gram = basis @ basis.T
    if not np.allclose(gram, np.eye(3), atol=2e-6):
        raise ValueError("This analysis expects the project's orthogonal HNEMD cell")
    return basis, lengths


def build_neighbours(species, positions, lattice):
    basis, lengths = orthogonal_basis(lattice)
    aligned = (positions @ basis.T) % lengths
    ge_indices = np.flatnonzero(species == "Ge")
    te_indices = np.flatnonzero(species == "Te")
    if ge_indices.size == 0 or te_indices.size == 0:
        raise ValueError("Both Ge and Te must be present")
    tree = cKDTree(aligned[te_indices], boxsize=lengths)
    distances, local_neighbours = tree.query(aligned[ge_indices], k=6)
    if float(np.max(distances)) > 3.8:
        raise ValueError(f"Six-neighbour cutoff exceeded: {np.max(distances):.3f} A")
    return ge_indices, te_indices[local_neighbours], basis, lengths, distances


def pjt_vectors(positions, ge_indices, neighbours, basis, lengths):
    aligned = (positions @ basis.T) % lengths
    ge = aligned[ge_indices]
    te = aligned[neighbours]
    displacement = te - ge[:, None, :]
    displacement -= np.rint(displacement / lengths) * lengths
    pjt_aligned = -np.mean(displacement, axis=1)
    return pjt_aligned @ basis


def fft_autocorrelation(values: np.ndarray, demean: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if demean:
        values = values - np.mean(values, axis=0, keepdims=True)
    nframes = values.shape[0]
    spectrum = np.fft.rfft(values, n=2 * nframes, axis=0)
    correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=2 * nframes, axis=0)[:nframes]
    correlation /= np.arange(nframes, 0, -1)[:, None]
    return np.sum(correlation, axis=1)


def positive_integral(time_ps: np.ndarray, correlation: np.ndarray) -> float:
    stop = len(correlation)
    nonpositive = np.flatnonzero(correlation[1:] <= 0)
    if nonpositive.size:
        stop = int(nonpositive[0] + 1)
    stop = max(stop, 2)
    return float(_trapezoid(correlation[:stop], time_ps[:stop]))


def write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(case: Path, keep_cache: bool) -> None:
    model_path = case / "model.xyz"
    dump_path = case / "dump.xyz"
    if not dump_path.is_file():
        raise FileNotFoundError(dump_path)

    species, model_positions, lattice = read_model(model_path)
    ge_indices, neighbours, basis, lengths, neighbour_distances = build_neighbours(
        species, model_positions, lattice
    )
    analysis_dir = case / "analysis"
    if analysis_dir.exists():
        raise FileExistsError(f"Analysis already exists: {analysis_dir}")
    analysis_dir.mkdir()
    cache_path = analysis_dir / "pjt_vectors.float32.dat"
    cache = np.memmap(
        cache_path,
        dtype=np.float32,
        mode="w+",
        shape=(EXPECTED_FRAMES, ge_indices.size, 3),
    )

    global_vectors = np.empty((EXPECTED_FRAMES, 3), dtype=np.float64)
    mean_amplitudes = np.empty(EXPECTED_FRAMES, dtype=np.float64)
    nematic = np.empty(EXPECTED_FRAMES, dtype=np.float64)
    frame_count = 0
    with dump_path.open("r", encoding="utf-8", errors="strict") as stream:
        while frame_count < EXPECTED_FRAMES:
            frame = read_xyz_frame(stream, expected_atoms=species.size)
            if frame is None:
                break
            frame_species, positions, header = frame
            if not np.array_equal(frame_species, species):
                raise ValueError(f"Atom ordering changed at frame {frame_count}")
            if "Lattice=" in header and not np.allclose(parse_lattice(header), lattice, atol=1e-6):
                raise ValueError("Cell changed during the fixed-cell NVE production")
            vectors = pjt_vectors(positions, ge_indices, neighbours, basis, lengths)
            cache[frame_count] = vectors.astype(np.float32)
            amplitudes = np.linalg.norm(vectors, axis=1)
            unit = vectors / np.maximum(amplitudes[:, None], 1e-12)
            global_vectors[frame_count] = np.mean(vectors, axis=0)
            mean_amplitudes[frame_count] = np.mean(amplitudes)
            polar_projection = unit @ basis[2]
            nematic[frame_count] = 0.5 * (3.0 * np.mean(polar_projection**2) - 1.0)
            frame_count += 1

    if frame_count < 100:
        raise ValueError(f"Only {frame_count} frames were found")
    if frame_count != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} frames, found {frame_count}")
    cache.flush()
    vectors = cache[:frame_count]
    global_vectors = global_vectors[:frame_count]
    mean_amplitudes = mean_amplitudes[:frame_count]
    nematic = nematic[:frame_count]
    time_ps = np.arange(frame_count) * DT_PS

    self_corr_sum = np.zeros(frame_count, dtype=np.float64)
    pjt_psd_sum = np.zeros(frame_count // 2 + 1, dtype=np.float64)
    count = 0
    window = np.hanning(frame_count)[:, None]
    for start in range(0, ge_indices.size, 256):
        block = np.asarray(vectors[:, start : start + 256], dtype=np.float64)
        amplitudes = np.linalg.norm(block, axis=2)
        unit = block / np.maximum(amplitudes[:, :, None], 1e-12)
        for component in range(3):
            self_corr_sum += fft_autocorrelation(unit[:, :, component], demean=False)
            fluctuation = block[:, :, component] - np.mean(block[:, :, component], axis=0, keepdims=True)
            fft = np.fft.rfft(fluctuation * window, axis=0)
            pjt_psd_sum += np.sum(np.abs(fft) ** 2, axis=1)
        count += block.shape[1]

    self_corr = self_corr_sum / count
    self_corr /= self_corr[0]
    tail_start = max(int(0.8 * frame_count), 1)
    plateau = float(np.median(self_corr[tail_start:]))
    connected = (self_corr - plateau) / max(1.0 - plateau, 1e-12)
    collective = fft_autocorrelation(global_vectors, demean=True)
    collective /= collective[0]

    frequencies = np.fft.rfftfreq(frame_count, d=DT_PS)
    pjt_psd = pjt_psd_sum / max(count * 3, 1)
    band = (frequencies >= 0) & (frequencies <= 5.0)
    area = float(_trapezoid(pjt_psd[band], frequencies[band]))
    pjt_psd_normalized = pjt_psd / area if area > 0 else pjt_psd
    low_band = (frequencies >= 0) & (frequencies <= 0.5)
    low_weight = float(_trapezoid(pjt_psd_normalized[low_band], frequencies[low_band]))
    peak_band = (frequencies > 0) & (frequencies <= 5.0)
    peak_frequency = float(frequencies[peak_band][np.argmax(pjt_psd_normalized[peak_band])])

    order_ratio_series = np.linalg.norm(global_vectors, axis=1) / np.maximum(mean_amplitudes, 1e-12)
    summary = {
        "temperature_K": int(next(line.split()[1] for line in (case / "run.in").read_text().splitlines() if line.strip().startswith("velocity"))),
        "replica": int(case.name.split("_")[-1]),
        "atoms": int(species.size),
        "ge_sites": int(ge_indices.size),
        "frames": int(frame_count),
        "dt_ps": DT_PS,
        "production_ps": float(frame_count * DT_PS),
        "mean_local_pjt_A": float(np.mean(mean_amplitudes)),
        "sem_local_pjt_A": float(np.std(mean_amplitudes, ddof=1) / math.sqrt(frame_count)),
        "mean_collective_order_ratio": float(np.mean(order_ratio_series)),
        "mean_nematic_order": float(np.mean(nematic)),
        "self_correlation_plateau": plateau,
        "connected_relaxation_time_ps": positive_integral(time_ps, connected),
        "collective_relaxation_time_ps": positive_integral(time_ps, collective),
        "low_frequency_weight_0_0p5_THz": low_weight,
        "pjt_spectral_peak_THz": peak_frequency,
        "max_initial_sixth_neighbour_A": float(np.max(neighbour_distances[:, -1])),
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    write_csv(
        analysis_dir / "correlation.csv",
        ("time_ps", "C_self_raw", "C_self_connected", "C_collective_connected"),
        (
            {
                "time_ps": f"{time_ps[i]:.8g}",
                "C_self_raw": f"{self_corr[i]:.10g}",
                "C_self_connected": f"{connected[i]:.10g}",
                "C_collective_connected": f"{collective[i]:.10g}",
            }
            for i in range(frame_count)
        ),
    )
    write_csv(
        analysis_dir / "spectrum.csv",
        ("frequency_THz", "pjt_spectral_density", "normalized_density"),
        (
            {
                "frequency_THz": f"{frequencies[i]:.8g}",
                "pjt_spectral_density": f"{pjt_psd[i]:.10g}",
                "normalized_density": f"{pjt_psd_normalized[i]:.10g}",
            }
            for i in range(frequencies.size)
        ),
    )
    write_csv(
        analysis_dir / "order_timeseries.csv",
        ("time_ps", "mean_local_pjt_A", "Mx_A", "My_A", "Mz_A", "order_ratio", "nematic_order"),
        (
            {
                "time_ps": f"{time_ps[i]:.8g}",
                "mean_local_pjt_A": f"{mean_amplitudes[i]:.10g}",
                "Mx_A": f"{global_vectors[i, 0]:.10g}",
                "My_A": f"{global_vectors[i, 1]:.10g}",
                "Mz_A": f"{global_vectors[i, 2]:.10g}",
                "order_ratio": f"{order_ratio_series[i]:.10g}",
                "nematic_order": f"{nematic[i]:.10g}",
            }
            for i in range(frame_count)
        ),
    )

    del vectors
    del cache
    if not keep_cache:
        cache_path.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path, help="trajectory directory containing model.xyz and dump.xyz")
    parser.add_argument("--keep-cache", action="store_true", help="retain the temporary vector memmap")
    parser.add_argument(
        "--validate-model-only",
        action="store_true",
        help="validate the GeTe6 neighbour definition without requiring dump.xyz",
    )
    args = parser.parse_args()
    case = args.case.resolve()
    if args.validate_model_only:
        species, positions, lattice = read_model(case / "model.xyz")
        ge_indices, _, _, _, distances = build_neighbours(species, positions, lattice)
        print(
            f"[PASS] {case}: {species.size} atoms, {ge_indices.size} Ge sites; "
            f"maximum sixth-neighbour distance = {np.max(distances[:, -1]):.4f} A"
        )
        return
    analyze(case, args.keep_cache)


if __name__ == "__main__":
    main()
