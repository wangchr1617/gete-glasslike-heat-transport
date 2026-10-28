# Data definitions

Units: temperature K; conductivity and its error W m⁻¹ K⁻¹; ordinary frequency
THz; time ps; displacement Å. `_A` denotes angstrom. Scientific definitions and
analysis protocols are given in the manuscript and Supplemental Material.

## HNEMD and PJT

| Field or file | Definition |
| --- | --- |
| `kappa_parallel`, `kappa_perpendicular` | κzz, κxx; κyy=κxx is assumed |
| `kappa_trace` | `(κparallel + 2κperpendicular)/3` |
| `anisotropy_ratio` | `κparallel/κperpendicular` |
| HNEMD `*_sem` | Sample SEM of ten 1 ns block means, `ddof=1` |
| `kappa.out` | Five columns: z-driven diagonal is column 5; x-driven diagonal is columns 1+2 |
| `mean_local_pjt_A` | Time-mean local amplitude, Å |
| `mean_collective_order_ratio` | Time mean of `|Q(t)|/A(t)`, dimensionless |
| `order_timeseries.csv` | Time, amplitude, collective-vector components and order parameters |
| `correlation.csv` | Raw/connected local and connected collective correlations |
| `*_relaxation_time_ps` | Initial positive-lobe integral, ending before the first non-positive sample |
| `spectrum.csv` | Local-coordinate power and density normalized over 0–5 THz |

PJT summary errors use SEM across independent replicas. Single-trajectory
`sem_` fields have a different sampling unit. Overlapping spectral windows
are not independent replicas. Legacy HNEMD errors are identified in
[archive notes](provenance.md).

## GKMA arrays

`04_D_modal_transport/processed/<T>K/modal_results.npz` contains:

| Array family | Shape / meaning |
| --- | --- |
| `metadata_json`, `schema_version`, `n_trajectories` | Sampling, units and schema |
| `trajectory_indices`, `input_sha256_json`, `input_line_count` | Replica IDs and source-file checks |
| `kappa_total_xyz_values` | `(10,3)`, replica-level endpoint conductivities |
| `kappa_diagonal_xyz_values`, `kappa_cross_xyz_values` | `(10,3)`, self/cross terms |
| `kappa_total_xyz_mean`, `kappa_total_xyz_sem` | `(3,)`, component means and SEMs |
| `kappa_total_scalar_values` | `(10,)`, replica trace means |
| `kappa_total_scalar_mean`, `kappa_total_scalar_sem` | Scalar mean and SEM |
| `kappa_mode_total_xyz_mean`, `kappa_mode_total_xyz_sem` | `(2058,3)`, modal contributions |
| `running_kappa_total_xyz_mean`, `running_kappa_total_xyz_sem` | `(3,1000)`, 0.05–50 ps integrals |
| `mode_sum_relative_error_values` | `(10,)`, numerical consistency checks |

`eigenvector.in` stores little-endian float32 signed squared angular
frequencies, followed by eigenvectors ordered mode → direction → atom.
Signed ordinary frequency is `sign(λ)*sqrt(abs(λ))/(2π)`; its sign labels
quadratic curvature. Frequency-density bins are 0.10 THz wide.

Componentwise or per-mode SEMs require covariance information to form scalar
or sector SEMs; do not treat them as independent errors.

## Supplemental arrays

| File | Layout / units |
| --- | --- |
| `q_displacement_spectra/*/transverse_spectra.npz` | q=(1,0,0), relative Ge–Te b/c components; all replicas retained |
| `time_ps`, `relative_transverse_amplitude` | `(5000,)`, `(5000,2)`; complex amplitudes in Å |
| `q_inverse_angstrom`, `reciprocal_supercell_indices`, `polarization_axes` | Wavevector and polarization metadata |
| `frequency_20ps_THz`, `frequency_50ps_THz` | `(501,)`, `(1251,)`; frequency grids |
| `transverse_psd_20ps`, `transverse_psd_50ps` | b+c power summed within each replica, Å²/THz |
| `segments_20ps`, `segments_50ps` | Hann-window counts, 50% overlap |
| `S2_vacf/*/mvac.out` | Time and Cartesian VACFs; 0.005 ps spacing |
| `S2_vacf/volume_outputs/*/*/dos.out` | Angular frequency in ps⁻¹ and Cartesian DOS; divide frequency by 2π for THz |
| `S3_effective_modes/*/fc2.hdf5`, `POSCAR` | Second-order force constants and primitive cells |
| `modal_descriptors/*/fc3.hdf5` | Third-order force constants |
| `modal_descriptors/*/kappa-m151515.hdf5` | `frequency`, `gamma[0]`: `(344,6)`; 15³ mesh |

`gamma` is a half-linewidth in ordinary THz; `tau_ps=1/(4*pi*gamma)`.
Group-velocity magnitudes are in THz Å (×100 for m/s). Descriptor plots retain
nonnegative frequencies; linewidth/lifetime plots also require positive gamma.
The VACF envelope uses Hilbert magnitude clipped to `[1e-8,1]` before `log10`.
Centroid tables use angular-frequency bands [3,6], [6,12], [12,20] ps⁻¹ and
relative volumes 0.99, 1, 1.01.
