#!/usr/bin/env python3
"""
postprocess_gkma.py — GKMA heatmode.out 后处理与验证 (v5, bin_size=1)

重要:
  - bin_size=1: 每个 bin 对应一个模式
  - J_total(t) = Σ_m J_m(t) → HCACF_total → κ_total
  - 标准 GKMA 单模贡献: ⟨J_m(t)·J_total(0)⟩
  - diagonal/cross 仅是数学自相关/交叉分解，不直接等同于 WTE POP/coherence
  - heatmode.out 5 分量: jxi, jxo, jyi, jyo, jz (面内/面外应力分解)
  - 单模热流: J_m = [jxi+jxo, jyi+jyo, jz]_m
  - 使用 GPUMD natural units；标量 κ = (κx+κy+κz)/3

输出:
  - traj_NN_results.npz (单轨迹 HCACF、模态贡献、metadata 和 SHA-256)
  - combined_results.npz (10 条轨迹的 mean/std/SEM)
  - summary.txt, validation_report.json, validation_report.txt

用法:
    python postprocess_gkma.py <温度点目录> [选项]
    --tcut PS:      HCACF 截断时间 ps (默认 50)
    --max-freq THz: 频率上限 THz (默认 10)
    --traj N:       仅处理第 N 条轨迹并保存中间结果
    --combine:      严格合并连续编号且 metadata 一致的单轨迹 npz
    --validate:     检查原始文件、有限值、模态自洽和 tcut 末段平台
    --force:        强制覆盖已有单轨迹结果
"""

import numpy as np
import argparse
import os
import sys
import glob
import time
import hashlib
import json

# numpy 2.0 兼容
if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
else:
    _trapz = np.trapz

# ── 物理常数 ──
KB = 8.617333262145e-5   # eV/K
# Match GPUMD's natural-unit constants in src/utilities/common.cuh.  Modal
# heat currents are emitted in GPUMD natural units, not in eV Å/fs.
TIME_UNIT_CONVERSION = 1.018051e1  # natural time -> fs
KAPPA_UNIT_CONVERSION = 1.573769e5  # natural conductivity -> W/(m K)


def detect_parameters(temp_dir):
    """从 traj_01/run.in 自动检测模拟参数."""
    run_file = os.path.join(temp_dir, "traj_01", "run.in")
    params = {}
    gkma_seen = False
    with open(run_file) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "time_step":
                params["time_step"] = float(parts[1])
            elif parts[0] == "compute_gkma":
                gkma_seen = True
                params["sample_interval"] = int(parts[1])
                params["first_mode"] = int(parts[2])
                params["last_mode"] = int(parts[3])
                if len(parts) > 5 and parts[4] == "bin_size":
                    params["bin_size"] = int(parts[5])
                elif len(parts) > 5 and parts[4] == "f_bin_size":
                    params["f_bin_size"] = float(parts[5])
            elif gkma_seen and parts[0] == "run":
                params["production_steps"] = int(parts[1])

    model_file = os.path.join(temp_dir, "model.xyz")
    if not os.path.isfile(model_file):
        model_file = os.path.join(temp_dir, "traj_01", "model.xyz")
    with open(model_file) as f:
        n_atom = int(f.readline().strip())
        lattice_line = f.readline().strip()
    params["n_atom"] = n_atom

    if 'Lattice="' in lattice_line:
        lat_str = lattice_line.split('Lattice="')[1].split('"')[0]
        lat = np.array([float(x) for x in lat_str.split()]).reshape(3, 3)
        params["volume"] = np.abs(np.linalg.det(lat))
    else:
        params["volume"] = None

    return params


def parse_temperature_from_dir(temp_dir):
    name = os.path.basename(os.fspath(temp_dir).rstrip("/\\"))
    return float(name.replace("K", ""))


class InputValidationError(ValueError):
    """Raised when a raw or intermediate GKMA input cannot be trusted."""


def _sha256_and_line_count(filepath, block_size=8 * 1024 * 1024):
    """Hash a text output and count physical lines in a single streaming pass."""
    digest = hashlib.sha256()
    line_count = 0
    last_byte = b""
    with open(filepath, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
            line_count += block.count(b"\n")
            last_byte = block[-1:]
    if os.path.getsize(filepath) and last_byte != b"\n":
        line_count += 1
    return digest.hexdigest(), line_count


def inspect_heatmode(filepath, n_bins, expected_steps=None):
    """Collect integrity facts without silently discarding any modal samples."""
    filepath = os.fspath(filepath)
    if not os.path.isfile(filepath):
        raise InputValidationError(f"heatmode.out not found: {filepath}")
    if n_bins < 1:
        raise InputValidationError(f"n_bins must be positive, got {n_bins}")

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        raise InputValidationError(f"heatmode.out is empty: {filepath}")
    sha256, n_lines = _sha256_and_line_count(filepath)
    remainder = n_lines % n_bins
    if remainder:
        raise InputValidationError(
            f"heatmode.out line count {n_lines} is not divisible by n_bins={n_bins}; "
            f"refusing to discard {remainder} incomplete modal rows"
        )
    n_steps = n_lines // n_bins
    if expected_steps is not None and n_steps != expected_steps:
        raise InputValidationError(
            f"heatmode.out has {n_steps} samples, expected {expected_steps}"
        )
    return {
        "path": os.path.abspath(filepath),
        "file_size_bytes": file_size,
        "sha256": sha256,
        "line_count": n_lines,
        "n_steps": n_steps,
        "n_bins": n_bins,
    }


def read_heatmode_chunked(filepath, n_bins, inspection=None, dtype=np.float32):
    """Read a validated heatmode file and reject malformed/non-finite values."""
    inspection = inspection or inspect_heatmode(filepath, n_bins)
    n_steps = inspection["n_steps"]
    print(f"  文件: {filepath} ({inspection['file_size_bytes']/1e9:.2f} GB)")
    print(f"  行数: {inspection['line_count']:,}, 时间步: {n_steps:,}, SHA256: {inspection['sha256'][:12]}…")
    try:
        data = np.empty((n_steps, n_bins, 5), dtype=dtype)
    except MemoryError as exc:
        est_gb = n_steps * n_bins * 5 * np.dtype(dtype).itemsize / 1e9
        raise InputValidationError(
            f"Unable to allocate {est_gb:.1f} GB for validated heatmode data"
        ) from exc

    import pandas as pd

    t0 = time.time()
    chunksize = n_bins * 50
    step = 0
    try:
        reader = pd.read_csv(
            filepath, sep=r"\s+", header=None, dtype=dtype,
            chunksize=chunksize, engine="c",
        )
        for chunk_index, chunk in enumerate(reader, start=1):
            values = chunk.to_numpy(copy=False)
            if values.shape[1] != 5:
                raise InputValidationError(
                    f"heatmode.out must have 5 columns, found {values.shape[1]}"
                )
            if len(values) % n_bins:
                raise InputValidationError(
                    f"Reader chunk {chunk_index} has an incomplete modal frame"
                )
            if not np.all(np.isfinite(values)):
                raise InputValidationError(
                    f"heatmode.out contains NaN/Inf in reader chunk {chunk_index}"
                )
            n_chunk_steps = len(values) // n_bins
            data[step:step + n_chunk_steps] = values.reshape(n_chunk_steps, n_bins, 5)
            step += n_chunk_steps
            if chunk_index % 20 == 0:
                print(f"  读取进度: {step:,}/{n_steps:,} ({step / n_steps * 100:.0f}%)")
    except InputValidationError:
        raise
    except Exception as exc:
        raise InputValidationError(f"Failed to parse heatmode.out: {exc}") from exc

    if step != n_steps:
        raise InputValidationError(f"Read {step} samples but inspected {n_steps}")
    elapsed = time.time() - t0
    print(f"  读取完成: {step:,} 步, 耗时 {elapsed:.1f}s")
    return data


def compute_hcacf_fft_vector(signal, axis=0):
    """FFT 法计算 HCACF (自动去均值, 归一化).

    HCACF(τ) = ⟨ΔJ(0)·ΔJ(τ)⟩, 其中 ΔJ = J − ⟨J⟩

    Args:
        signal: (...n_steps..., ...modes...) 沿 axis 的输入信号
        axis: 时间轴

    Returns:
        hcacf: 同 shape 的 HCACF
    """
    n_steps = signal.shape[axis]

    # 去均值
    signal = signal - signal.mean(axis=axis, keepdims=True)

    # FFT → PSD → IFFT (zero-padding to 2×n)
    fft = np.fft.rfft(signal, n=2 * n_steps, axis=axis)
    psd = np.abs(fft) ** 2
    corr = np.fft.irfft(psd, axis=axis)

    # 截取前 n_steps
    slc = [slice(None)] * signal.ndim
    slc[axis] = slice(None, n_steps)
    corr = corr[tuple(slc)]

    # 归一化: /(n_steps - tau)
    norm_shape = [1] * signal.ndim
    norm_shape[axis] = n_steps
    norm = np.arange(n_steps, 0, -1, dtype=np.float64).reshape(norm_shape)
    corr = corr / norm

    return corr


def scalar_kappa(kappa_xyz):
    """Return the isotropic scalar conductivity (the mean diagonal value)."""
    kappa_xyz = np.asarray(kappa_xyz, dtype=np.float64)
    if kappa_xyz.shape != (3,):
        raise ValueError(f"Expected three directional conductivities, got {kappa_xyz.shape}")
    return float(np.mean(kappa_xyz))


def _validate_hcacf_inputs(hcacf_xyz, dt_natural, volume, temperature, tcut_ps):
    hcacf_xyz = np.asarray(hcacf_xyz, dtype=np.float64)
    if hcacf_xyz.ndim != 2 or hcacf_xyz.shape[0] != 3:
        raise ValueError("hcacf_xyz must have shape (3, n_correlation_steps)")
    if hcacf_xyz.shape[1] < 2:
        raise ValueError("At least two correlation steps are required")
    if not np.all(np.isfinite(hcacf_xyz)):
        raise ValueError("HCACF contains non-finite values")
    if dt_natural <= 0 or volume <= 0 or temperature <= 0 or tcut_ps <= 0:
        raise ValueError("dt_natural, volume, temperature, and tcut_ps must be positive")
    return hcacf_xyz


def _cutoff_steps(n_steps, dt_natural, tcut_ps):
    """Return the number of stored samples inside a physical cutoff time."""
    requested_steps = int(np.floor(tcut_ps * 1000.0 / TIME_UNIT_CONVERSION / dt_natural)) + 1
    return min(n_steps, max(2, requested_steps))


def _hann_window(n_steps):
    """Cosine taper equal to one at t=0 and zero at the final sample."""
    if n_steps == 1:
        return np.ones(1, dtype=np.float64)
    index = np.arange(n_steps, dtype=np.float64)
    return 0.5 * (1.0 + np.cos(np.pi * index / (n_steps - 1)))


def _kappa_prefactor(volume, temperature):
    return KAPPA_UNIT_CONVERSION / (float(volume) * KB * float(temperature) ** 2)


def integrate_hcacf(hcacf_xyz, dt_natural, volume, temperature, tcut_ps):
    """Integrate directional HCACFs in GPUMD natural units.

    Args:
        hcacf_xyz: Array shaped ``(3, n_lags)`` in GPUMD heat-current units.
        dt_natural: Correlation sampling interval in GPUMD natural time.
        volume: Simulation-cell volume in Å^3.
        temperature: Temperature in K.
        tcut_ps: Requested cutoff in ps.

    Returns:
        ``(running_kappa_xyz, kappa_xyz)`` where the first array has shape
        ``(3, n_cut)`` and the final array has shape ``(3,)``, both in W/(m K).
    """
    hcacf_xyz = _validate_hcacf_inputs(hcacf_xyz, dt_natural, volume, temperature, tcut_ps)
    n_cut = _cutoff_steps(hcacf_xyz.shape[1], dt_natural, tcut_ps)
    weighted = hcacf_xyz[:, :n_cut] * _hann_window(n_cut)
    prefactor = _kappa_prefactor(volume, temperature)

    running = np.zeros_like(weighted)
    running[:, 1:] = np.cumsum(
        0.5 * (weighted[:, :-1] + weighted[:, 1:]), axis=1
    ) * dt_natural * prefactor
    return running, running[:, -1].copy()


def spectral_kappa(hcacf_xyz, dt_natural, volume, temperature, tcut_ps, max_freq_thz):
    """Return the cosine-transformed directional conductivity spectrum.

    Frequency zero is retained explicitly so that ``spectrum_xyz[:, 0]`` is
    exactly the same Green-Kubo integral returned by :func:`integrate_hcacf`.
    """
    hcacf_xyz = _validate_hcacf_inputs(hcacf_xyz, dt_natural, volume, temperature, tcut_ps)
    if max_freq_thz <= 0:
        raise ValueError("max_freq_thz must be positive")

    n_cut = _cutoff_steps(hcacf_xyz.shape[1], dt_natural, tcut_ps)
    weighted = hcacf_xyz[:, :n_cut] * _hann_window(n_cut)
    n_frequency = int(np.floor(max_freq_thz * tcut_ps)) + 1
    frequency_thz = np.arange(n_frequency, dtype=np.float64) / tcut_ps
    time_ps = np.arange(n_cut, dtype=np.float64) * dt_natural * TIME_UNIT_CONVERSION / 1000.0
    cosine = np.cos(2.0 * np.pi * np.outer(frequency_thz, time_ps))
    spectrum_xyz = _kappa_prefactor(volume, temperature) * _trapz(
        weighted[:, np.newaxis, :] * cosine[np.newaxis, :, :],
        dx=dt_natural,
        axis=2,
    )
    return frequency_thz, spectrum_xyz


def _unbiased_cross_correlation(reference, signal, n_steps):
    """Return <reference(0) signal(t)> for three directional signals."""
    fft_reference = np.fft.rfft(reference, n=2 * n_steps, axis=0)
    fft_signal = np.fft.rfft(signal, n=2 * n_steps, axis=0)
    correlation = np.fft.irfft(
        np.conj(fft_reference) * fft_signal, n=2 * n_steps, axis=0
    )[:n_steps]
    return correlation / np.arange(n_steps, 0, -1, dtype=np.float64)[:, np.newaxis]


def process_trajectory(hm_file, n_bins, dt_fs, volume, temperature, tcut_ps, max_freq_thz,
                       expected_steps=None):
    """Process one trajectory using GPUMD units and standard GKMA correlations."""
    inspection = inspect_heatmode(hm_file, n_bins, expected_steps=expected_steps)
    data = read_heatmode_chunked(hm_file, n_bins, inspection=inspection)
    n_steps, n_modes, _ = data.shape
    if n_steps < 4:
        raise InputValidationError(f"At least four samples are required, found {n_steps}")
    print(f"  Data shape: {data.shape} (steps × modes × 5)")

    t0 = time.time()
    j_mode = np.empty((n_steps, n_modes, 3), dtype=np.float64)
    j_mode[:, :, 0] = data[:, :, 0] + data[:, :, 1]
    j_mode[:, :, 1] = data[:, :, 2] + data[:, :, 3]
    j_mode[:, :, 2] = data[:, :, 4]
    del data
    print(f"  Modal current assembled: {j_mode.nbytes / 1e9:.2f} GB in {time.time() - t0:.1f}s")

    sample_dt_natural = float(dt_fs) / TIME_UNIT_CONVERSION
    j_total = j_mode.sum(axis=1)
    total_centered = j_total - j_total.mean(axis=0, keepdims=True)
    hcacf_total_xyz = compute_hcacf_fft_vector(total_centered, axis=0).T
    running_total_xyz, kappa_total_xyz = integrate_hcacf(
        hcacf_total_xyz, sample_dt_natural, volume, temperature, tcut_ps
    )
    frequency_thz, kappa_total_spectrum_xyz = spectral_kappa(
        hcacf_total_xyz, sample_dt_natural, volume, temperature, tcut_ps, max_freq_thz
    )

    hcacf_diagonal_xyz = np.zeros((3, n_steps), dtype=np.float64)
    kappa_mode_total_xyz = np.empty((n_modes, 3), dtype=np.float64)
    norm = 1.0 / np.arange(n_steps, 0, -1, dtype=np.float64)
    fft_total = np.fft.rfft(total_centered, n=2 * n_steps, axis=0)
    report_interval = max(1, n_modes // 10)
    t0 = time.time()
    for mode in range(n_modes):
        mode_signal = j_mode[:, mode, :]
        mode_centered = mode_signal - mode_signal.mean(axis=0, keepdims=True)
        fft_mode = np.fft.rfft(mode_centered, n=2 * n_steps, axis=0)
        cross = np.fft.irfft(
            np.conj(fft_total) * fft_mode, n=2 * n_steps, axis=0
        )[:n_steps] * norm[:, np.newaxis]
        _running_mode, kappa_mode_total_xyz[mode] = integrate_hcacf(
            cross.T, sample_dt_natural, volume, temperature, tcut_ps
        )
        auto = np.fft.irfft(np.abs(fft_mode) ** 2, n=2 * n_steps, axis=0)[:n_steps]
        hcacf_diagonal_xyz += (auto * norm[:, np.newaxis]).T
        if (mode + 1) % report_interval == 0:
            elapsed = time.time() - t0
            eta = elapsed / (mode + 1) * (n_modes - mode - 1)
            print(f"  Mode {mode + 1}/{n_modes} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    _running_diagonal_xyz, kappa_diagonal_xyz = integrate_hcacf(
        hcacf_diagonal_xyz, sample_dt_natural, volume, temperature, tcut_ps
    )
    _freq_check, kappa_diagonal_spectrum_xyz = spectral_kappa(
        hcacf_diagonal_xyz, sample_dt_natural, volume, temperature, tcut_ps, max_freq_thz
    )
    kappa_cross_xyz = kappa_total_xyz - kappa_diagonal_xyz
    kappa_cross_spectrum_xyz = kappa_total_spectrum_xyz - kappa_diagonal_spectrum_xyz
    mode_sum_xyz = kappa_mode_total_xyz.sum(axis=0)
    denominator = max(float(np.linalg.norm(kappa_total_xyz)), np.finfo(np.float64).tiny)
    mode_sum_relative_error = float(np.linalg.norm(mode_sum_xyz - kappa_total_xyz) / denominator)
    del j_mode

    return {
        "inspection": inspection,
        "sample_dt_fs": float(dt_fs),
        "sample_dt_natural": sample_dt_natural,
        "hcacf_total_xyz": hcacf_total_xyz,
        "hcacf_diagonal_xyz": hcacf_diagonal_xyz,
        "running_kappa_total_xyz": running_total_xyz,
        "kappa_total_xyz": kappa_total_xyz,
        "kappa_diagonal_xyz": kappa_diagonal_xyz,
        "kappa_cross_xyz": kappa_cross_xyz,
        "kappa_mode_total_xyz": kappa_mode_total_xyz,
        "kappa_total_spectrum_xyz": kappa_total_spectrum_xyz,
        "kappa_diagonal_spectrum_xyz": kappa_diagonal_spectrum_xyz,
        "kappa_cross_spectrum_xyz": kappa_cross_spectrum_xyz,
        "frequency_thz": frequency_thz,
        "kappa_total_scalar": scalar_kappa(kappa_total_xyz),
        "kappa_diagonal_scalar": scalar_kappa(kappa_diagonal_xyz),
        "kappa_cross_scalar": scalar_kappa(kappa_cross_xyz),
        "kappa_mode_total_scalar": np.mean(kappa_mode_total_xyz, axis=1),
        "mode_sum_relative_error": mode_sum_relative_error,
    }


RESULT_SCHEMA_VERSION = 5
RESULT_ARRAY_KEYS = (
    "hcacf_total_xyz", "hcacf_diagonal_xyz", "running_kappa_total_xyz",
    "kappa_total_xyz", "kappa_diagonal_xyz", "kappa_cross_xyz",
    "kappa_mode_total_xyz", "frequency_thz", "kappa_total_spectrum_xyz",
    "kappa_diagonal_spectrum_xyz", "kappa_cross_spectrum_xyz",
    "mode_sum_relative_error",
)


def _metadata_json(metadata):
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def _build_result_payload(result, metadata):
    payload = {
        "schema_version": np.array(RESULT_SCHEMA_VERSION),
        "metadata_json": np.array(_metadata_json(metadata)),
        "input_sha256": np.array(result["inspection"]["sha256"]),
        "input_line_count": np.array(result["inspection"]["line_count"]),
    }
    for key in RESULT_ARRAY_KEYS:
        payload[key] = np.asarray(result[key])
    return payload


def save_result_atomic(filepath, result, metadata):
    """Persist a trajectory result only after a read-back integrity check."""
    filepath = os.fspath(filepath)
    payload = _build_result_payload(result, metadata)
    temporary_path = f"{filepath}.tmp.npz"
    try:
        np.savez_compressed(temporary_path, **payload)
        load_result_checked(temporary_path)
        os.replace(temporary_path, filepath)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_result_checked(filepath):
    """Load an NPZ only if it has the v5 schema, finite arrays, and metadata."""
    try:
        with np.load(filepath, allow_pickle=False) as archive:
            required = {"schema_version", "metadata_json", "input_sha256", "input_line_count", *RESULT_ARRAY_KEYS}
            missing = required.difference(archive.files)
            if missing:
                raise InputValidationError(f"Result file missing keys: {sorted(missing)}")
            schema_version = int(np.asarray(archive["schema_version"]).item())
            if schema_version != RESULT_SCHEMA_VERSION:
                raise InputValidationError(
                    f"Result schema {schema_version} is not supported (expected {RESULT_SCHEMA_VERSION})"
                )
            try:
                metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise InputValidationError("Result metadata_json is invalid") from exc
            result = {key: np.asarray(archive[key]) for key in RESULT_ARRAY_KEYS}
            result["input_sha256"] = str(np.asarray(archive["input_sha256"]).item())
            result["input_line_count"] = int(np.asarray(archive["input_line_count"]).item())
            result["metadata"] = metadata
    except InputValidationError:
        raise
    except Exception as exc:
        raise InputValidationError(f"Cannot load result file {filepath}: {exc}") from exc

    for key, value in result.items():
        if isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
            raise InputValidationError(f"Result key {key} contains NaN/Inf")
    if result["kappa_total_xyz"].shape != (3,):
        raise InputValidationError("kappa_total_xyz must have shape (3,)")
    if result["kappa_mode_total_xyz"].ndim != 2 or result["kappa_mode_total_xyz"].shape[1] != 3:
        raise InputValidationError("kappa_mode_total_xyz must have shape (n_modes, 3)")
    return result


def evaluate_tcut_convergence(hcacf_xyz, dt_natural, volume, temperature,
                              tcut_grid_ps, relative_tolerance, plateau_points=3):
    """Evaluate conductivity stability over the final contiguous tcut points."""
    grid = np.asarray(tcut_grid_ps, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError("tcut_grid_ps must contain at least two increasing positive values")
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    if not isinstance(plateau_points, (int, np.integer)) or not 2 <= plateau_points <= len(grid):
        raise ValueError("plateau_points must be an integer between 2 and the tcut-grid length")
    values_xyz = np.array([
        integrate_hcacf(hcacf_xyz, dt_natural, volume, temperature, float(tcut))[1]
        for tcut in grid
    ])
    values_scalar = np.array([scalar_kappa(value) for value in values_xyz])
    reference = values_scalar[-1]
    denominator = max(abs(reference), np.finfo(np.float64).tiny)
    relative_change = np.abs(values_scalar - reference) / denominator
    plateau_start = len(grid) - plateau_points
    plateau_relative_change = relative_change[plateau_start:]
    return {
        "tcut_grid_ps": grid,
        "values_WmK": values_xyz,
        "scalar_values_WmK": values_scalar,
        "relative_change_to_final": relative_change,
        "relative_tolerance": float(relative_tolerance),
        "plateau_points": int(plateau_points),
        "plateau_tcut_ps": grid[plateau_start:],
        "plateau_relative_change_to_final": plateau_relative_change,
        "pass": bool(np.all(plateau_relative_change <= relative_tolerance)),
    }


def _trajectory_result_files(temp_dir):
    indexed = []
    for filepath in glob.glob(os.path.join(temp_dir, "traj_*_results.npz")):
        stem = os.path.basename(filepath).replace("_results.npz", "")
        try:
            indexed.append((int(stem.split("_")[1]), filepath))
        except (IndexError, ValueError):
            continue
    return sorted(indexed)


def combine_trajectories(temp_dir, expected_count=10):
    """Combine exactly the requested number of metadata-identical trajectories."""
    indexed_files = _trajectory_result_files(temp_dir)
    expected_indices = list(range(1, expected_count + 1))
    actual_indices = [index for index, _ in indexed_files]
    if actual_indices != expected_indices:
        raise InputValidationError(
            f"Expected result files for trajectories {expected_indices}, found {actual_indices}"
        )
    loaded = [load_result_checked(filepath) for _, filepath in indexed_files]
    reference_metadata = _metadata_json(loaded[0]["metadata"])
    for (index, _), result in zip(indexed_files[1:], loaded[1:]):
        if _metadata_json(result["metadata"]) != reference_metadata:
            raise InputValidationError(f"metadata mismatch in traj_{index:02d}_results.npz")

    scalar_keys = ("kappa_total_xyz", "kappa_diagonal_xyz", "kappa_cross_xyz")
    array_keys = (
        "kappa_mode_total_xyz", "frequency_thz", "kappa_total_spectrum_xyz",
        "kappa_diagonal_spectrum_xyz", "kappa_cross_spectrum_xyz",
        "hcacf_total_xyz", "hcacf_diagonal_xyz", "running_kappa_total_xyz",
    )
    combined = {
        "schema_version": np.array(RESULT_SCHEMA_VERSION),
        "metadata_json": np.array(reference_metadata),
        "trajectory_indices": np.array(actual_indices, dtype=np.int64),
        "input_sha256_json": np.array(json.dumps([item["input_sha256"] for item in loaded])),
        "input_line_count": np.array([item["input_line_count"] for item in loaded], dtype=np.int64),
    }
    for key in scalar_keys + array_keys:
        stack = np.stack([item[key] for item in loaded], axis=0)
        combined[f"{key}_mean"] = np.mean(stack, axis=0)
        combined[f"{key}_std"] = np.std(stack, axis=0, ddof=1)
        combined[f"{key}_sem"] = combined[f"{key}_std"] / np.sqrt(expected_count)
        if key in scalar_keys:
            combined[f"{key}_values"] = stack
    scalar_values = np.array([scalar_kappa(item["kappa_total_xyz"]) for item in loaded])
    combined["kappa_total_scalar_values"] = scalar_values
    combined["kappa_total_scalar_mean"] = np.array(np.mean(scalar_values))
    combined["kappa_total_scalar_std"] = np.array(np.std(scalar_values, ddof=1))
    combined["kappa_total_scalar_sem"] = np.array(np.std(scalar_values, ddof=1) / np.sqrt(expected_count))
    combined["n_trajectories"] = np.array(expected_count)
    combined["mode_sum_relative_error_values"] = np.array([item["mode_sum_relative_error"] for item in loaded])

    output_path = os.path.join(temp_dir, "combined_results.npz")
    temporary_path = f"{output_path}.tmp.npz"
    try:
        np.savez_compressed(temporary_path, **combined)
        with np.load(temporary_path, allow_pickle=False) as check:
            if int(check["schema_version"].item()) != RESULT_SCHEMA_VERSION:
                raise InputValidationError("Combined result read-back failed")
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)

    summary = (
        "GKMA Post-Processing Summary (schema v5)\n"
        "========================================\n"
        f"Temperature:      {loaded[0]['metadata']['temperature_K']:.0f} K\n"
        f"Trajectories:     {expected_count}\n"
        f"HCACF cutoff:     {loaded[0]['metadata']['tcut_ps']:.3f} ps\n"
        f"κ scalar:         {combined['kappa_total_scalar_mean'].item():.4f} ± "
        f"{combined['kappa_total_scalar_sem'].item():.4f} W/(m K)\n"
        "Definition:       (κ_x + κ_y + κ_z) / 3\n"
    )
    with open(os.path.join(temp_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(summary)
    print(summary)
    return combined


def _load_combined_checked(filepath):
    try:
        with np.load(filepath, allow_pickle=False) as archive:
            required = {
                "schema_version", "metadata_json", "trajectory_indices", "kappa_total_scalar_mean",
                "kappa_total_scalar_sem", "n_trajectories", "hcacf_total_xyz_mean",
                "mode_sum_relative_error_values",
            }
            missing = required.difference(archive.files)
            if missing:
                raise InputValidationError(f"Combined result missing keys: {sorted(missing)}")
            if int(archive["schema_version"].item()) != RESULT_SCHEMA_VERSION:
                raise InputValidationError("Unsupported combined result schema")
            result = {key: np.asarray(archive[key]) for key in archive.files}
            result["metadata"] = json.loads(str(result["metadata_json"].item()))
    except InputValidationError:
        raise
    except Exception as exc:
        raise InputValidationError(f"Cannot load combined result {filepath}: {exc}") from exc
    for key, value in result.items():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise InputValidationError(f"Combined key {key} contains NaN/Inf")
    return result


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _read_gkmd_reference(csv_path, temperature):
    if not csv_path or not os.path.isfile(csv_path):
        return {"available": False}
    import csv
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(line for line in handle if not line.startswith("#")):
            if float(row["T_K"]) == float(temperature):
                return {
                    "available": True,
                    "temperature_K": float(temperature),
                    "kappa_band_sum_WmK": float(row["kappa_band_sum"]),
                    "kappa_total_fft_WmK": float(row["kappa_total_fft"]),
                }
    return {"available": False, "reason": f"No {temperature:g} K row"}


def validate_temperature(temp_dir, gkmd_reference=None, expected_count=10,
                         tcut_grid_ps=(10.0, 20.0, 30.0, 40.0, 50.0),
                         tcut_relative_tolerance=0.05, tcut_plateau_points=3):
    """Validate raw inputs, trajectory outputs, and the final tcut plateau."""
    report = {"temperature_directory": os.path.abspath(temp_dir), "issues": []}
    try:
        temperature = parse_temperature_from_dir(temp_dir)
        params = detect_parameters(temp_dir)
        n_modes = params["last_mode"] - params["first_mode"] + 1
        expected_steps = params["production_steps"] // params["sample_interval"]
        sample_dt_fs = params["time_step"] * params["sample_interval"]
        indexed_files = _trajectory_result_files(temp_dir)
        expected_indices = list(range(1, expected_count + 1))
        if [index for index, _ in indexed_files] != expected_indices:
            raise InputValidationError("Result files do not cover all expected trajectories")
        raw_checks = []
        trajectory_results = []
        for index, result_path in indexed_files:
            result = load_result_checked(result_path)
            raw_path = os.path.join(temp_dir, f"traj_{index:02d}", "heatmode.out")
            inspection = inspect_heatmode(raw_path, n_modes, expected_steps=expected_steps)
            if inspection["sha256"] != result["input_sha256"]:
                raise InputValidationError(f"traj_{index:02d}: raw SHA256 differs from saved result")
            if inspection["line_count"] != result["input_line_count"]:
                raise InputValidationError(f"traj_{index:02d}: raw line count differs from saved result")
            if float(result["mode_sum_relative_error"]) > 1e-8:
                raise InputValidationError(f"traj_{index:02d}: mode contribution sum is not self-consistent")
            raw_checks.append({"trajectory": index, **inspection, "mode_sum_relative_error": float(result["mode_sum_relative_error"])})
            trajectory_results.append(result)

        combined = _load_combined_checked(os.path.join(temp_dir, "combined_results.npz"))
        if list(combined["trajectory_indices"].astype(int)) != expected_indices:
            raise InputValidationError("combined_results.npz trajectory indices do not match required set")
        recomputed_scalar = np.array([scalar_kappa(item["kappa_total_xyz"]) for item in trajectory_results])
        if not np.allclose(recomputed_scalar.mean(), combined["kappa_total_scalar_mean"].item(), rtol=1e-12):
            raise InputValidationError("combined scalar mean does not reproduce trajectory results")
        convergence = evaluate_tcut_convergence(
            combined["hcacf_total_xyz_mean"], sample_dt_fs / TIME_UNIT_CONVERSION,
            params["volume"], temperature, tcut_grid_ps, tcut_relative_tolerance,
            plateau_points=tcut_plateau_points,
        )
        gkmd = _read_gkmd_reference(gkmd_reference, temperature)
        report.update({
            "temperature_K": temperature,
            "raw_checks": raw_checks,
            "convergence": convergence,
            "gkmd_reference": gkmd,
            "overall_pass": bool(convergence["pass"]),
        })
    except Exception as exc:
        report["issues"].append(str(exc))
        report["overall_pass"] = False

    json_report = _json_ready(report)
    with open(os.path.join(temp_dir, "validation_report.json"), "w", encoding="utf-8") as handle:
        json.dump(json_report, handle, indent=2, ensure_ascii=False)
    text_lines = [
        "GKMA Validation Report",
        "======================",
        f"Overall pass: {json_report['overall_pass']}",
    ]
    if json_report["issues"]:
        text_lines.extend(["Issues:", *[f"- {issue}" for issue in json_report["issues"]]])
    elif "convergence" in json_report:
        convergence = json_report["convergence"]
        plateau = convergence["plateau_tcut_ps"]
        text_lines.append(
            f"tcut plateau: {plateau[0]:.3f}–{plateau[-1]:.3f} ps "
            f"({convergence['plateau_points']} points)"
        )
        text_lines.append(f"tcut convergence pass: {json_report['convergence']['pass']}")
    with open(os.path.join(temp_dir, "validation_report.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(text_lines) + "\n")
    return json_report


def process_temperature(temp_dir, tcut_ps=50.0, max_freq_thz=10.0,
                        traj_only=None, force=False, expected_count=10):
    """Process a temperature directory or combine its validated trajectory results."""
    temperature = parse_temperature_from_dir(temp_dir)
    params = detect_parameters(temp_dir)
    required = ("time_step", "sample_interval", "volume", "production_steps")
    missing = [key for key in required if key not in params or params[key] is None]
    if missing:
        raise InputValidationError(f"run.in/model.xyz is missing required parameters: {missing}")
    if params.get("bin_size") != 1:
        raise InputValidationError("This v5 workflow requires compute_gkma ... bin_size 1")
    n_modes = params["last_mode"] - params["first_mode"] + 1
    n_bins = n_modes
    sample_dt_fs = params["time_step"] * params["sample_interval"]
    expected_steps = params["production_steps"] // params["sample_interval"]
    if params["production_steps"] % params["sample_interval"]:
        raise InputValidationError("Production run length is not divisible by GKMA sample interval")
    print(f"T={temperature:.0f} K | modes={n_modes} | sample_dt={sample_dt_fs:.3f} fs | "
          f"expected samples={expected_steps}")

    if traj_only == "combine":
        return combine_trajectories(temp_dir, expected_count=expected_count)

    if traj_only is not None:
        trajectory_indices = [int(traj_only)]
    else:
        trajectory_indices = list(range(1, expected_count + 1))

    for index in trajectory_indices:
        trajectory_dir = os.path.join(temp_dir, f"traj_{index:02d}")
        heatmode_path = os.path.join(trajectory_dir, "heatmode.out")
        result_path = os.path.join(temp_dir, f"traj_{index:02d}_results.npz")
        if os.path.exists(result_path) and not force:
            load_result_checked(result_path)
            print(f"traj_{index:02d}: validated result exists; skipping (use --force to recompute)")
            continue
        result = process_trajectory(
            heatmode_path, n_bins=n_bins, dt_fs=sample_dt_fs, volume=params["volume"],
            temperature=temperature, tcut_ps=tcut_ps, max_freq_thz=max_freq_thz,
            expected_steps=expected_steps,
        )
        metadata = {
            "temperature_K": temperature,
            "sample_dt_fs": sample_dt_fs,
            "volume_A3": float(params["volume"]),
            "n_atoms": int(params["n_atom"]),
            "first_mode": int(params["first_mode"]),
            "last_mode": int(params["last_mode"]),
            "n_modes": int(n_modes),
            "n_bins": int(n_bins),
            "tcut_ps": float(tcut_ps),
            "max_freq_thz": float(max_freq_thz),
        }
        save_result_atomic(result_path, result, metadata)
        print(f"traj_{index:02d}: κ_scalar={result['kappa_total_scalar']:.4f} W/(m K), "
              f"mode-sum error={result['mode_sum_relative_error']:.3e}")
    if traj_only is None:
        return combine_trajectories(temp_dir, expected_count=expected_count)
    return None


def _parse_tcut_grid(value):
    try:
        grid = tuple(float(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tcut grid must be comma-separated positive numbers") from exc
    if len(grid) < 2 or any(item <= 0 for item in grid) or any(b <= a for a, b in zip(grid, grid[1:])):
        raise argparse.ArgumentTypeError("tcut grid must contain at least two increasing positive values")
    return grid


def _default_gkmd_reference_path():
    workdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(workdir, "gkmd", "kappa_omega_data_v2.csv")


def main():
    parser = argparse.ArgumentParser(
        description="GKMA heatmode.out 后处理与验证 (v5, GPUMD natural units)")
    parser.add_argument("temp_dir", help="温度点目录路径")
    parser.add_argument("--tcut", type=float, default=50.0,
                        help="HCACF 最大相关时间 ps (默认 50)")
    parser.add_argument("--max-freq", type=float, default=10.0,
                        help="频率上限 THz (默认 10)")
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--traj", type=int, default=None,
                              help="仅处理指定编号轨迹")
    action_group.add_argument("--combine", action="store_true",
                              help="合并已验证的单轨迹 NPZ")
    action_group.add_argument("--validate", action="store_true",
                              help="检查原始数据、有限值、模态自洽和 tcut 末段平台")
    parser.add_argument("--force", action="store_true",
                        help="强制覆盖已有结果")
    parser.add_argument("--expected-trajectories", type=int, default=10,
                        help="要求的连续轨迹数 (默认 10)")
    parser.add_argument("--gkmd-reference", default=None,
                        help="GK-MD kappa_omega_data_v2.csv 路径（仅记录，不参与验证）")
    parser.add_argument("--tcut-grid", type=_parse_tcut_grid,
                        default=(10.0, 20.0, 30.0, 40.0, 50.0),
                        help="验证用递增 tcut 网格，单位 ps，例如 10,20,30,40,50")
    parser.add_argument("--tcut-relative-tolerance", type=float, default=0.05,
                        help="tcut 收敛相对阈值 (默认 0.05)")
    parser.add_argument("--tcut-plateau-points", type=int, default=3,
                        help="末段平台包含的最后 tcut 点数 (默认 3)")
    args = parser.parse_args()

    if args.expected_trajectories < 1:
        parser.error("--expected-trajectories must be positive")
    if args.tcut_relative_tolerance <= 0:
        parser.error("--tcut-relative-tolerance must be positive")
    if not 2 <= args.tcut_plateau_points <= len(args.tcut_grid):
        parser.error("--tcut-plateau-points must be between 2 and the tcut-grid length")
    default_gkmd = None
    try:
        if args.validate:
            report = validate_temperature(
                args.temp_dir,
                gkmd_reference=args.gkmd_reference or default_gkmd,
                expected_count=args.expected_trajectories,
                tcut_grid_ps=args.tcut_grid,
                tcut_relative_tolerance=args.tcut_relative_tolerance,
                tcut_plateau_points=args.tcut_plateau_points,
            )
            print(f"Validation overall_pass={report['overall_pass']}")
            return 0 if report["overall_pass"] else 2
        trajectory_mode = "combine" if args.combine else args.traj
        process_temperature(
            args.temp_dir,
            tcut_ps=args.tcut,
            max_freq_thz=args.max_freq,
            traj_only=trajectory_mode,
            force=args.force,
            expected_count=args.expected_trajectories,
        )
        return 0
    except (InputValidationError, MemoryError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
