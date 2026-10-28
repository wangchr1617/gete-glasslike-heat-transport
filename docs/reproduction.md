# Running the calculations

Run the following commands from the repository root. Use a dedicated Python
3.12 environment; outputs default to `minimal_reproduction/outputs/`.

## Tables and figures

```sh
python -m pip install -r minimal_reproduction/requirements.txt
python minimal_reproduction/rebuild_tables.py
python minimal_reproduction/reproduce_figures.py
```

Use `--only 3 S3 S6` to select figures, `--formats png pdf svg` for additional
exports, or `--output-dir minimal_reproduction/outputs/recheck` for a new output
root. These options apply to `reproduce_figures.py`. Dependency snapshots are
in `requirements-locked.txt` and `requirements-linewidths-locked.txt`.

## GPUMD runs

```sh
python minimal_reproduction/prepare_case.py --list
python minimal_reproduction/prepare_case.py --case A/585K/para
cd minimal_reproduction/outputs/runs/A/585K/para
gpumd
```

The assembler copies the model, potential, run control and any required
eigenvectors into a new directory. Existing destinations are refused. Case
prefixes are `A`, `B`, `C`, `D` and `VACF`; use `--case all` to assemble all cases.
GPUMD needs CUDA and support for the commands in the archived `run.in` files.
The original GPUMD commit/CUDA build is not recorded.

## New trajectory postprocessing

After running the corresponding prepared cases:

```sh
python minimal_reproduction/postprocess/pjt.py minimal_reproduction/outputs/runs/B/585K/traj_01
python minimal_reproduction/postprocess/gkma.py minimal_reproduction/outputs/runs/D/585K --traj 1
python minimal_reproduction/postprocess/gkma.py minimal_reproduction/outputs/runs/D/585K --combine
python minimal_reproduction/postprocess/gkma.py minimal_reproduction/outputs/runs/D/585K --validate
python minimal_reproduction/postprocess/displacement.py --case all --source-root minimal_reproduction/outputs/runs --output-dir minimal_reproduction/outputs/new_displacement
```

For GKMA, process each replica with `--traj N` before combining. The PJT command
also accepts C cases. Raw position and modal heat-current files are not bundled;
new production runs require several GB per trajectory, with GKMA heat-current
files around 12 GB each.

## Force constants and eigenvectors

```sh
python -m pip install -r minimal_reproduction/requirements-force-constants.txt
python minimal_reproduction/force_constants/prepare.py --temperature 585 --output-dir minimal_reproduction/outputs/fc_585K
```

Run GPUMD in each generated `force_eval/structure_XXXX/` directory, then:

```sh
python minimal_reproduction/force_constants/extract_forces.py minimal_reproduction/outputs/fc_585K
python minimal_reproduction/force_constants/extract_ifcs.py minimal_reproduction/outputs/fc_585K --n-jobs 1
python minimal_reproduction/force_constants/convert_fc_to_hdf5.py minimal_reproduction/outputs/fc_585K
python minimal_reproduction/force_constants/convert_eigenvectors.py 05_supplemental/S3_effective_modes/585K/fc2.hdf5 04_D_modal_transport/inputs/585K/model.xyz minimal_reproduction/outputs/eigenvector_585K.in
```

All generated force files are required for fitting. Use archived eigenvectors
with archived modal data; a regenerated basis belongs with a new GKMA run.
Fitting requires substantial RAM. Parameters are defined in the scripts and
described in the Supplemental Material.

For three-phonon calculations, install `requirements-linewidths.txt` in a
**separate** Python 3.12 environment: its phonopy version differs from the
figure environment. Then:

```sh
python minimal_reproduction/force_constants/linewidths.py --temperature 300 --output-dir minimal_reproduction/outputs/linewidth_check_300K --check-inputs
python minimal_reproduction/force_constants/linewidths.py --temperature 300 --output-dir minimal_reproduction/outputs/linewidth_rerun_300K
```

`--check-inputs` loads the inputs without running scattering calculations.
Additional utilities for cell construction and VACF centroids are in
`force_constants/` and `postprocess/`.

See [data definitions](data_dictionary.md) and [archive notes](provenance.md).
