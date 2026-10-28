#!/usr/bin/env python3
"""PRB Fig. 4: effective modes, GKMA sectors, and MD conductivity agreement.

Panels (a)-(c) use nine temperature-specific quadratic projection bases.
Panel (d) compares HNEMD and GKMA on their common nine-temperature grid.
"""

from __future__ import annotations

from pathlib import Path

from _figure_style import output_path, save_figure
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figure_style import (apply_style, temperature_color, TEMPERATURE_CMAP,
                           TEMPERATURE_NORM, participation_cmap)
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
GKMA = ROOT / "04_D_modal_transport" / "processed"
WTE_RTA = ROOT / "05_supplemental" / "S3_effective_modes"
HNEMD_CSV = ROOT / "01_A_tensor_transport" / "hnemd_tensor_source.csv"
OUTPUT = output_path("fig_04_modal_reorganization")
TEMPERATURES = np.array([300, 400, 500, 550, 585, 600, 650, 700, 800])
TC = 585
N_MODES = 2058
BIN_WIDTH = 0.10
BIN_EDGES = np.arange(-1.8, 5.6 + BIN_WIDTH, BIN_WIDTH)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
COLORS = {
    "navy": "#33539E",
    "sky": "#6094C3",
    "lavender": "#8877B1",
    "grey": "#999999",
}

apply_style()


def effective_gamma_frequencies() -> tuple[np.ndarray, np.ndarray]:
    import phonopy
    from phonopy.interface.vasp import read_vasp

    positive, negative_magnitude = [], []
    for temperature in TEMPERATURES:
        directory = WTE_RTA / f"{temperature}K"
        phonon = phonopy.load(
            unitcell=read_vasp(directory / "POSCAR"),
            supercell_matrix=np.diag([7, 7, 7]),
            primitive_matrix=np.eye(3),
            force_constants_filename=directory / "fc2.hdf5",
            produce_fc=False,
        )
        phonon.run_qpoints([[0, 0, 0]])
        frequency = np.asarray(phonon.qpoints.frequencies[0])
        positive.append(frequency[3])
        negative_magnitude.append(abs(frequency[0]))
    return np.asarray(positive), np.asarray(negative_magnitude)


def read_modes(temperature: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(GKMA / f"{temperature}K" / "modal_results.npz") as result:
        kappa_mode = result["kappa_mode_total_xyz_mean"].mean(axis=1)
    omega_squared = np.fromfile(
        ROOT / "04_D_modal_transport" / "inputs" / f"{temperature}K" / "eigenvector.in", dtype=np.float32, count=N_MODES
    )
    if omega_squared.size != N_MODES or kappa_mode.size != N_MODES:
        raise ValueError(f"Unexpected number of GKMA modes at {temperature} K")
    frequency = np.sign(omega_squared) * np.sqrt(np.abs(omega_squared)) / (2 * np.pi)
    return frequency, kappa_mode


def transport_on_common_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hnemd = pd.read_csv(HNEMD_CSV).set_index("T")
    if not hnemd.index.is_unique or not set(TEMPERATURES).issubset(hnemd.index):
        raise ValueError("HNEMD source must contain each common-grid temperature once")
    hnemd = hnemd.loc[TEMPERATURES]
    hnemd_value = ((hnemd["kappa_para"] + 2 * hnemd["kappa_perp"]) / 3).to_numpy()

    # The source CSV used population-normalized block errors. The manuscript's
    # Table S1 reports sample SEM for ten 1 ns blocks (nine degrees of freedom).
    sample_sem_factor = np.sqrt(10 / 9)
    hnemd_sem = (
        np.sqrt(hnemd["kappa_para_err"] ** 2 + 4 * hnemd["kappa_perp_err"] ** 2)
        / 3
        * sample_sem_factor
    ).to_numpy()

    gkma_value, gkma_sem = [], []
    for temperature in TEMPERATURES:
        with np.load(GKMA / f"{temperature}K" / "modal_results.npz") as result:
            values = result["kappa_total_scalar_values"]
            if values.size != 10:
                raise ValueError(f"Expected ten GKMA trajectories at {temperature} K")
            gkma_value.append(values.mean())
            gkma_sem.append(values.std(ddof=1) / np.sqrt(values.size))
    return hnemd_value, hnemd_sem, np.asarray(gkma_value), np.asarray(gkma_sem)


positive, negative = effective_gamma_frequencies()
spectra, sectors = [], []
for temperature in TEMPERATURES:
    frequency, kappa_mode = read_modes(int(temperature))
    spectra.append(np.histogram(frequency, bins=BIN_EDGES, weights=kappa_mode)[0] / BIN_WIDTH)
    sectors.append(
        [
            kappa_mode[frequency < 0].sum(),
            kappa_mode[(frequency >= 0) & (frequency < 2)].sum(),
            kappa_mode[frequency >= 2].sum(),
        ]
    )
spectra = np.asarray(spectra)
sectors = np.asarray(sectors)
hnemd_value, hnemd_sem, gkma_value, gkma_sem = transport_on_common_grid()



assert np.isclose(hnemd_value[TEMPERATURES == 585][0], 0.857, atol=0.001)
assert np.isclose(gkma_value[TEMPERATURES == 585][0], 0.790, atol=0.001)
assert np.isclose(sectors[TEMPERATURES == 585, 0][0], 0.088, atol=0.001)

fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2))
ax_a, ax_b, ax_c, ax_d = axes.ravel()
# Equal rectangular panels fill the grid without square-box shrinkage.
# The outer right margin accommodates the colourbar independently of the gutter.
fig.subplots_adjust(left=0.105, right=0.905, bottom=0.115, top=0.95,
                    wspace=0.20, hspace=0.36)

# (a) Effective Γ-point curvatures.
ax_a.axvline(TC, color=COLORS["grey"], linestyle="--", linewidth=1, alpha=0.7)
ax_a.plot(
    TEMPERATURES, positive, "o-", color=COLORS["navy"], linewidth=1.4,
    markersize=4.7, markerfacecolor="white", markeredgewidth=1.2,
    label="Positive optical",
)
ax_a.plot(
    TEMPERATURES, negative, "s-", color=COLORS["sky"], linewidth=1.4,
    markersize=4.7, markerfacecolor="white", markeredgewidth=1.2,
    label="|Negative|",
)
ax_a.set(xlim=(280, 820), ylim=(0.2, 3.6), xlabel="Temperature (K)", ylabel="Γ-point effective frequency (THz)")
ax_a.legend(loc="upper right", fontsize=7.2, frameon=True, framealpha=0.88,
            edgecolor="#CCCCCC", handlelength=1.25, labelspacing=0.35)

# (b) Temperature-matched signed-frequency density. The nine fitted
# temperatures are shown as discrete rows; no intermediate temperature is inferred.
scale = np.nanpercentile(np.abs(spectra), 98)
mesh = ax_b.pcolormesh(
    BIN_CENTERS, TEMPERATURES, spectra, cmap="PuOr", shading="nearest",
    rasterized=True, norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale),
)
ax_b.axvline(0, color=COLORS["grey"], linestyle="--", linewidth=0.9, alpha=0.8)
ax_b.axvline(2, color=COLORS["grey"], linestyle=":", linewidth=0.9, alpha=0.8)
ax_b.axhline(TC, color=COLORS["grey"], linestyle="--", linewidth=0.9, alpha=0.8)
ax_b.set(xlim=(BIN_EDGES[0], BIN_EDGES[-1]), xlabel="Signed effective frequency (THz)", ylabel="Temperature (K)")
cax = inset_axes(
    ax_b, width="3%", height="100%", loc="lower left",
    bbox_to_anchor=(1.025, 0, 1, 1), bbox_transform=ax_b.transAxes,
    borderpad=0,
)
cbar = fig.colorbar(mesh, cax=cax)
cbar.ax.set_title(r"$\kappa_\nu$", fontsize=8.2, pad=5)
cbar.ax.tick_params(direction="in", length=3, width=0.6, labelsize=7)

# (c) Sector-integrated projected heat current.
ax_c.axvline(TC, color=COLORS["grey"], linestyle="--", linewidth=1, alpha=0.7)
ax_c.axhline(0, color=COLORS["grey"], linestyle="--", linewidth=0.8, alpha=0.7)
for index, (label, color, marker) in enumerate(
    [
        ("ν < 0", COLORS["lavender"], "D"),
        ("0 ≤ ν < 2 THz", COLORS["navy"], "o"),
        ("ν ≥ 2 THz", COLORS["sky"], "s"),
    ]
):
    ax_c.plot(
        TEMPERATURES, sectors[:, index], marker=marker, color=color,
        linewidth=1.4, markersize=4.7, markerfacecolor="white",
        markeredgewidth=1.2, label=label,
    )
ax_c.set(xlim=(280, 820), ylim=(-0.04, 1.05), xlabel="Temperature (K)",
         ylabel=r"Projected $\kappa$ (W m$^{-1}$ K$^{-1}$)")
ax_c.legend(loc="upper right", fontsize=7.2, frameon=True, framealpha=0.88,
            edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.35)

# (d) Matched HNEMD and GKMA totals. The dense HNEMD transition grid remains
# in Fig. 1; this panel uses only temperatures shared with GKMA.
ax_d.axvline(TC, color=COLORS["grey"], linestyle="--", linewidth=1, alpha=0.7)
for value, sem, marker, color, label in [
    (hnemd_value, hnemd_sem, "s", COLORS["sky"], "HNEMD"),
    (gkma_value, gkma_sem, "o", COLORS["navy"], "GKMA"),
]:
    ax_d.errorbar(
        TEMPERATURES, value, yerr=sem, fmt=marker + "-", color=color,
        linewidth=1.4, markersize=4.7, markerfacecolor="white",
        markeredgewidth=1.2, elinewidth=0.9, capsize=2.5,
        label=label,
    )
ax_d.set(xlim=(280, 820), ylim=(0, 2.45), xlabel="Temperature (K)",
         ylabel=r"Tensor-trace $\kappa$ (W m$^{-1}$ K$^{-1}$)")
ax_d.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.88,
            edgecolor="#CCCCCC", handlelength=1.2, labelspacing=0.35)

for label, ax in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
    ax.set_title(f"({label})", loc="center", fontsize=9, fontweight="bold", pad=4)

save_figure(fig, OUTPUT.with_suffix(".png"), dpi=600, facecolor="white")
plt.close(fig)
print(f"Saved {OUTPUT}.png")
