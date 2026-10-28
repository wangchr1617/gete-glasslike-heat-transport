#!/usr/bin/env python3
"""Fig. S3: transverse relative-displacement spectra at reciprocal-supercell q=(1,0,0).

Curves and SEM use all 18 independent trajectories, grouped as 3/6/3/3/3.
Power in b and c is summed within each replica before estimating SEM.
The 20 ps Hann-window spectrum uses 50% overlap and no area normalization.
The archive also retains the 50 ps spectra and coherent amplitudes for checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _figure_style import output_path, save_figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figure_style import (apply_style, temperature_color, TEMPERATURE_CMAP,
                           TEMPERATURE_NORM, participation_cmap)
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "05_supplemental" / "q_displacement_spectra"
GROUPS = {
    "native_550": (3, "550 K", temperature_color(550), "-"),
    "native_585": (6, "585 K", temperature_color(585), "--"),
    "native_650": (3, "650 K", temperature_color(650), "-."),
    "cubic_585": (3, "Cubic", "#6094C3", "-."),
    "rhombo_585": (3, "Rhombohedral", "#33539E", "-"),
}
COMPARISONS = (
    ("native_550", "native_585", "native_650"),
    ("rhombo_585", "native_585", "cubic_585"),
)
WINDOWS = (20,)


def load_data(source):
    groups = {}
    qnorms = []
    for group, (n, *_rest) in GROUPS.items():
        paths = sorted(source.glob(f"{group}_r*/transverse_spectra.npz"))
        expected = {f"{group}_r{i:02d}" for i in range(1, n + 1)}
        if {p.parent.name for p in paths} != expected:
            raise ValueError(f"Expected {n} distinct replicas for {group}")
        spectra = {20: []}
        frequencies = {}
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                assert z["time_ps"].shape == (5000,)
                assert np.allclose(np.diff(z["time_ps"]), 0.02)
                assert np.array_equal(z["reciprocal_supercell_indices"], [1, 0, 0])
                q = z["q_inverse_angstrom"]
                axes = z["polarization_axes"]
                assert np.allclose(axes @ axes.T, np.eye(3), atol=2e-6)
                assert np.allclose(axes[1:] @ q, 0, atol=1e-8)
                qnorms.append(np.linalg.norm(q))
                for window in WINDOWS:
                    f = z[f"frequency_{window}ps_THz"]
                    s = z[f"transverse_psd_{window}ps"]
                    assert s.shape == (len(f),)
                    assert np.isfinite(s).all() and (s >= 0).all()
                    assert np.allclose(np.diff(f), 1 / window)
                    if window in frequencies:
                        assert np.allclose(f, frequencies[window])
                    frequencies[window] = f.copy()
                    spectra[window].append(s)
        groups[group] = {
            "f": frequencies,
            "s": {w: np.stack(values) for w, values in spectra.items()},
        }
    return groups, (min(qnorms), max(qnorms))


def mean_sem(values):
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def style():
    apply_style()


def make_figure(source, output):
    groups, qrange = load_data(source)
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.20, top=0.87,
                        wspace=0.29)
    titles = ("(a)", "(b)")
    max_spectrum = 0
    for col, keys in enumerate(COMPARISONS):
        ax = axes[col]
        for key, marker in zip(keys, ("o", "s", "D")):
            n, label, color, linestyle = GROUPS[key]
            if col == 1 and key == "native_585":
                label = "Native"
            item = groups[key]
            f = item["f"][20]
            m, se = mean_sem(item["s"][20])
            display = f <= 1.5 + 1e-8
            max_spectrum = max(max_spectrum, (m + se)[display].max())
            ax.fill_between(f[display], (m-se)[display], (m+se)[display],
                            color=color, alpha=0.18, linewidth=0)
            # Mark every original frequency bin; no interpolation or smoothing.
            ax.plot(f[display], m[display], color=color, ls=linestyle, lw=1.1,
                    marker=marker, markersize=2.3, markerfacecolor="white",
                    markeredgewidth=0.55,
                    label=label)
        ax.set_xlim(0, 1.5)
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel(r"$S_{\mathrm{rel},b}+S_{\mathrm{rel},c}$ (Å$^2$ THz$^{-1}$)")
        ax.legend(loc="upper right", handlelength=2.3, labelspacing=0.45)

    for ax, title in zip(axes, titles):
        ax.set_ylim(0, np.ceil(max_spectrum * 1.1))
        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    save_figure(fig, output, dpi=600, facecolor="white",
                metadata={"Description": __doc__, "Software": "Matplotlib"})
    plt.close(fig)
    print(f"Saved {output.name}: 600 dpi PNG; all 18 trajectories used.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    make_figure(args.data_dir, output_path("fig_s03_collective_displacement_spectra.png"))
