#!/usr/bin/env python3
"""Read-only trajectory audit and q-resolved displacement spectra (not velocity SED)."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MASSES = {'Ge': 72.630, 'Te': 127.603}


def frame(f):
    line = f.readline()
    if not line:
        return None
    n = int(line.strip())
    header = f.readline()
    if 'Properties=species:S:1:pos:R:3' not in header:
        raise ValueError('Expected species/position-only XYZ property layout')
    prop = re.search(r'Properties=(\S+)', header).group(1)
    if prop != 'species:S:1:pos:R:3':
        raise ValueError('Additional properties require an explicit parser update')
    cell = np.fromstring(re.search(r'Lattice="([^"]+)"', header).group(1), sep=' ').reshape(3, 3)
    if not re.search(r'pbc="T T T"', header):
        raise ValueError('Three periodic directions required')
    rows = [f.readline().split() for _ in range(n)]
    if any(len(row) != 4 for row in rows):
        raise ValueError('Truncated or malformed atom block')
    species = np.array([r[0] for r in rows])
    pos = np.array([r[1:] for r in rows], dtype=float)
    if not np.isfinite(pos).all():
        raise ValueError('Nonfinite positions')
    t = re.search(r'Time=([^\s]+)', header)
    return species, pos, cell, None if t is None else float(t.group(1))


def welch(x, dt, window_ps):
    """Symmetrized +/-q, one-sided PSD; complex spatial projections allowed."""
    n = round(window_ps / dt)
    if n < 4 or len(x) < n:
        raise ValueError('Insufficient frames for spectral window')
    w = np.hanning(n)
    total = []
    for start in range(0, len(x) - n + 1, n // 2):
        y = x[start:start+n]
        y = y - y.mean(axis=0)
        ft = np.fft.fft(y * w.reshape((-1,) + (1,) * (x.ndim-1)), axis=0)
        k = np.arange(n//2+1)
        ps = (np.abs(ft[k])**2 + np.abs(ft[(-k) % n])**2) * dt / (2*np.sum(w*w))
        ps[1:-1 if n % 2 == 0 else None] *= 2
        total.append(ps)
    return np.fft.rfftfreq(n, dt), np.mean(total, axis=0), len(total)


def cases():
    for temp, count in [(550, 3), (585, 6), (650, 3)]:
        for i in range(1, count+1):
            yield f'native_{temp}_r{i:02}', f'B/{temp}K/traj_{i:02}'
    for metric in ['cubic', 'rhombo']:
        for i in range(1, 4):
            yield f'{metric}_585_r{i:02}', f'C/585K/{metric}_metric/pjt/traj_{i:02}'


def analyze(name, rel, args):
    src = args.source_root / rel
    out = args.output_dir / name
    if out.exists():
        raise FileExistsError(f'{out} exists; archive it before rerunning')
    with (src/'model.xyz').open() as f:
        sp, ref, cell, _ = frame(f)
    if len(sp) != 23040 or set(sp) != set(MASSES):
        raise ValueError('Expected 23040 Ge/Te atoms')
    inv = np.linalg.inv(cell)
    axes = cell / np.linalg.norm(cell, axis=1)[:, None]
    if not np.allclose(axes @ axes.T, np.eye(3), atol=2e-6):
        raise ValueError('This workflow requires the existing orthogonal supercells')
    # Integer reciprocal-supercell indices: exactly compatible with periodicity.
    h = np.array([[0,0,0], [1,0,0], [2,0,0], [3,0,0], [0,0,1], [0,0,2], [0,0,3]])
    q = 2*np.pi*h @ inv.T
    phase = np.exp(-2j*np.pi*(ref @ inv) @ h.T)
    masks = [sp == s for s in ['Ge', 'Te']]
    weights = np.array([MASSES[s] for s in sp])
    prev = None
    unwrapped = None
    times, amplitudes = [], []
    max_jump = 0.
    max_rms = 0.
    with (src/'dump.xyz').open() as f:
        while True:
            data = frame(f)
            if data is None:
                break
            s, pos, c, t = data
            if not np.array_equal(s, sp) or not np.allclose(c, cell, atol=1e-7, rtol=0):
                raise ValueError('Atom ordering/species or cell changed')
            if t is None or (times and abs(t-times[-1]-20.) > 0.01):
                raise ValueError('Missing, repeated, or nonuniform timestamps (expected 20 fs)')
            frac = pos @ inv
            delta = frac - (ref @ inv if prev is None else prev)
            delta -= np.rint(delta)
            step = delta @ cell
            if prev is not None:
                max_jump = max(max_jump, float(np.max(np.linalg.norm(step, axis=1))))
                if max_jump > 3.:
                    raise ValueError('Frame-to-frame jump >3 A: inspect mapping, continuity or diffusion')
            unwrapped = step if unwrapped is None else unwrapped + step
            prev = frac
            u = unwrapped - np.average(unwrapped, weights=weights, axis=0)
            max_rms = max(max_rms, float(np.sqrt(np.mean(np.sum(u*u, axis=1)))))
            # Species coherent displacement, normalized by sqrt(number of species atoms).
            a = np.stack([phase[m].T @ u[m] / np.sqrt(m.sum()) for m in masks], axis=1)
            amplitudes.append(a @ axes.T)
            times.append(t)
            if len(times) % 500 == 0:
                print(name, len(times), 'frames', flush=True)
    if len(times) != args.expected_frames:
        raise ValueError(f'Expected {args.expected_frames} frames, got {len(times)}')
    x = np.array(amplitudes)
    # Four channels: Ge, Te, in-phase sum, and relative Ge-Te displacement.
    x = np.concatenate([x, ((x[:,:,0]+x[:,:,1])/np.sqrt(2))[:,:,None],
                        ((x[:,:,0]-x[:,:,1])/np.sqrt(2))[:,:,None]], axis=2)
    dt = float(np.mean(np.diff(times))) / 1000
    spectra = {}
    for window in [20., 50.]:
        freq, psd, nseg = welch(x, dt, window)
        spectra[f'frequency_{int(window)}ps_THz'] = freq
        spectra[f'psd_{int(window)}ps'] = psd
        spectra[f'segments_{int(window)}ps'] = nseg
    out.mkdir(parents=True)
    np.savez_compressed(out/'displacement_spectra.npz', time_ps=np.array(times)/1000,
                        amplitude=x, q_inverse_angstrom=q, reciprocal_supercell_indices=h,
                        polarization_axes=axes, **spectra)
    report = dict(case=name, frames=len(times), atoms=len(sp),
                  sample_interval_fs=dt*1000, first_time_fs=times[0], last_time_fs=times[-1],
                  max_step_displacement_A=max_jump, max_displacement_rms_A=max_rms,
                  model_sha256=hashlib.sha256((src/'model.xyz').read_bytes()).hexdigest(),
                  trajectory_bytes=(src/'dump.xyz').stat().st_size,
                  channels=['Ge','Te','in_phase','relative'], components=['cell_a','cell_b','cell_c'],
                  psd_units='angstrom^2/THz', psd_axes=['frequency','q','channel','component'],
                  interpretation='Coherent displacement projections; NOT branch eigenmode spectra, velocity SED, or heat-current spectra',
                  status='file_checks_passed; physical stationarity and mode mapping require review')
    (out/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('DONE', name, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument('--case', default='native_585_r01', help='Case name, all, or list')
    p.add_argument('--expected-frames', type=int, default=5000)
    args = p.parse_args()
    available = dict(cases())
    if args.case == 'list':
        print(json.dumps(available, indent=2)); return
    if args.case != 'all' and args.case not in available:
        p.error('Unknown case; use --case list')
    for name, rel in available.items():
        if args.case in ['all', name]:
            analyze(name, rel, args)


if __name__ == '__main__':
    main()
