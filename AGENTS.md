# Repository conventions

This repository follows Results A–D and Supplemental Material text sections
S1–S5, which contain six supplemental figures. Use the bilingual README's figure
map: legacy supplemental directory prefixes are not the current figure numbers.

Keep public documentation concise: file contents, commands and necessary data
definitions. Scientific results and their interpretation belong in the manuscript.

Keep the published numerical data and simulation inputs intact when adding new
analyses. Place new results in separate files, label units in table headers,
and update the README when the file map changes.

Reproduction programs live in `minimal_reproduction/`; generated output and
local environments are ignored. Assemble runs in new directories instead of
modifying archived input folders. Changes to input comments must preserve all
effective GPUMD commands, parameters and seeds.

Keep the legacy fixed-metric conductivity CSV. Its sample-SEM successor has a
separate filename. Preserve the archived GKMA 50 ps endpoint estimator and
document its distinction from a 30–50 ps average in `docs/provenance.md`.

Keep one-off verification scripts and audit records local. Before publishing,
check numerical consistency, effective input commands and the final diff.
