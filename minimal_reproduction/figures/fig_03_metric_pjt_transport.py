#!/usr/bin/env python3
"""Main Fig. 3: conductivity and PJT alignment under imposed cell metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from _figure_style import output_path, save_figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _figure_style import (apply_style, temperature_color, TEMPERATURE_CMAP,
                           TEMPERATURE_NORM, participation_cmap)
import numpy as np
import seaborn as sns


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
HNEMD = ROOT / "03_C_fixed_metric"
PJT = ROOT / "02_B_pjt_dynamics" / "replicas"
OUTPUT = output_path("fig_03_metric_pjt_transport.png")

COLORS = {
    "navy": "#33539E", "sky": "#6094C3", "lavender": "#8877B1",
    "light_lavender": "#CFC8E2", "grey": "#999999",
}
CONDITIONS = {
    550: (("Native", "native_metric"), ("Cubic", "cubic_metric")),
    585: (("Rhombohedral", "rhombo_metric"), ("Native", "native_metric"),
          ("Cubic", "cubic_metric")),
    650: (("Native", "native_metric"), ("Rhombohedral", "rhombo_metric")),
}
SERIES = (
    ("kappa_perp", "kappa_perp_sem", -0.13, "s", COLORS["navy"], r"$\kappa_\perp$"),
    ("kappa_para", "kappa_para_sem", +0.13, "o", COLORS["sky"], r"$\kappa_\parallel$"),
    ("kappa_trace", "kappa_trace_sem", 0.0, "D", COLORS["lavender"], r"$\kappa_{\rm tr}$"),
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


transport = {(int(row["temperature_K"]), row["metric"]): row
             for row in read_csv(HNEMD / "fixed_metric_conductivity_sample_sem.csv")}
assert set(transport) == {(temperature, metric)
                          for temperature, conditions in CONDITIONS.items()
                          for _, metric in conditions}


def order_values(temperature: int, metric: str) -> np.ndarray:
    n = 6 if metric == "native_metric" and temperature == 585 else 3
    if metric == "native_metric":
        base = PJT / f"{temperature}K"
    else:
        base = HNEMD / "pjt_replicas" / f"{temperature}K" / metric
    values = []
    for replica in range(1, n + 1):
        path = (base / f"traj_{replica:02d}" / "summary.json" if metric == "native_metric"
                else base / f"traj_{replica:02d}.json")
        item = json.loads(path.read_text(encoding="utf-8"))
        if item["temperature_K"] != temperature or item["replica"] != replica:
            raise ValueError(f"PJT identity mismatch: {path}")
        values.append(float(item["mean_collective_order_ratio"]))
    return np.asarray(values)


sns.set_style("ticks")
sns.set_context("paper", font_scale=1.15)
apply_style()

fig = plt.figure(figsize=(7.0, 4.0))
grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.0), width_ratios=(1.0, 1.35, 1.0),
                        left=0.095, right=0.995, bottom=0.155, top=0.92,
                        hspace=0.24, wspace=0.18)
axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(2)])

for column, (temperature, conditions) in enumerate(CONDITIONS.items()):
    upper, lower = axes[:, column]
    x = np.arange(len(conditions), dtype=float)
    for field, spread, offset, marker, color, label in SERIES:
        values = []
        errors = []
        for _, metric in conditions:
            item = transport[(temperature, metric)]
            if field == "kappa_trace":
                values.append((float(item["kappa_para"]) + 2 * float(item["kappa_perp"])) / 3)
                errors.append(np.hypot(float(item["kappa_para_sem"]),
                                       2 * float(item["kappa_perp_sem"])) / 3)
            else:
                values.append(float(item[field]))
                errors.append(float(item[spread]))
        upper.errorbar(x + offset, values, yerr=errors, color=color, marker=marker,
                       linestyle="none", markerfacecolor="white", markeredgewidth=1.1,
                       markersize=5.0, capsize=2.2, capthick=0.8, elinewidth=0.8,
                       label=label if column == 2 else None, zorder=3)
    upper.set_title(f"({chr(97 + column)})  {temperature} K", fontsize=9,
                    fontweight="bold", pad=4)
    upper.set_ylim(0.68, 1.035)
    upper.set_yticks((0.7, 0.8, 0.9, 1.0))
    upper.set_xticks(x, labels=[])
    upper.set_xlim(-0.42, len(conditions) - 0.58)
    if column == 0:
        upper.set_ylabel(r"$\kappa$ (W m$^{-1}$ K$^{-1}$)", labelpad=4)
    else:
        upper.tick_params(labelleft=False)
    if column == 2:
        upper.legend(loc="upper right", fontsize=7.2, frameon=False,
                     borderaxespad=0.35, handletextpad=0.35)

    for position, (_, metric) in enumerate(conditions):
        values = order_values(temperature, metric)
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(len(values)))
        jitter = np.linspace(-0.09, 0.09, len(values))
        run_artist = lower.scatter(position + jitter, values, s=11, marker="o",
                                   facecolor=COLORS["light_lavender"], edgecolor="white",
                                   linewidth=0.3, zorder=3)
        mean_artist = lower.errorbar(position, mean, yerr=sem, linestyle="none", marker="s",
                                     markersize=5.8, markerfacecolor="white",
                                     markeredgecolor=COLORS["lavender"], markeredgewidth=1.15,
                                     color=COLORS["lavender"], capsize=2.5, capthick=0.9,
                                     elinewidth=0.9, zorder=4)
    lower.set_ylim(0, 0.53)
    lower.set_yticks((0, 0.1, 0.2, 0.3, 0.4, 0.5))
    lower.set_xlim(-0.42, len(conditions) - 0.58)
    lower.set_xticks(x, labels=[label for label, _ in conditions])
    lower.tick_params(axis="x", pad=4)
    lower.set_title(f"({chr(100 + column)})", fontsize=9,
                    fontweight="bold", pad=4)
    if column == 0:
        lower.set_ylabel(r"PJT order ratio $R$", labelpad=4)
    else:
        lower.tick_params(labelleft=False)
    if column == 2:
        lower.legend(handles=[run_artist, mean_artist],
                     labels=["Independent runs", "Mean ± SEM"],
                     loc="upper right", fontsize=7.2, frameon=False,
           borderaxespad=0.35, handletextpad=0.35)

save_figure(fig, OUTPUT, dpi=600, facecolor="white",
            edgecolor="none", bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved {OUTPUT}")
