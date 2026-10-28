# Archive notes

Original numerical files and effective simulation controls are preserved.
Reference PNGs match the supplied final DOCXs; reconstructed figures are saved
separately.

## Estimators

- **Fixed-cell errors:** `fixed_metric_conductivity.csv` is the legacy export.
  Use `fixed_metric_conductivity_sample_sem.csv` for consistent sample SEM.
  The native-control conversion factor is `sqrt(10/9)` for ten blocks.
  Fig. 3 replot errors can therefore differ from the reference PNG.
- **GKMA:** archived totals and sectors use the 50 ps integration endpoint.
  The manuscript Methods describes a 30–50 ps average. The table builder
  exports both total estimators under separate names and retains the archived
  sectors.

Supplemental directory prefixes retain the historical archive order.
The current figure map is in [05_supplemental/README.md](../05_supplemental/README.md).
