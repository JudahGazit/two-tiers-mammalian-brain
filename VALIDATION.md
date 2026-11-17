# Validation status — 15 October 2026

Validated using the existing Windows / CPython 3.13.6 research environment.
Exact package versions are recorded in `REFERENCE_ENVIRONMENT.json` and
`requirements-lock.txt`. A fresh environment installation was not performed.

Completed checks:

- All 22 exported research source files match their recorded source/export
  SHA-256 hashes.
- Comparing parsed Python syntax trees after reversing documented path changes
  confirms the scientific expressions are unchanged. The only additional source
  adjustment is the optional FFmpeg executable location. Whitespace and LF line
  endings are normalized in the export.
- All included Python files parse and compile successfully.
- The synthetic demo runs from an unrelated temporary directory without data,
  artwork or access to the private repository. It verifies mean unweighted
  shortest-path length independently against NetworkX and checks identity
  projection numerically. Seed 0 gives 40 nodes, 12 core nodes and mean shortest
  path 1.6935897435897436.
- Missing data produces an explicit input diagnostic and unsuccessful exit.
- Direct dependency versions match the recorded reference environment.
- Coarse input checks pass for all nine figure commands with the existing
  external data directory. This does not establish complete cohort coverage.
- `figure1` completed with exit code 0 and wrote `Figures/figure1.pdf` (1,237,483
  bytes). Its runtime was not retained in the run log.
- `supfigure2` completed with exit code 0 in 15.169 seconds and wrote
  `SuppFigures/figure2_3.png` (666,745 bytes).

Generated manuscript images and PDFs remain local in ignored `outputs/`; they
are not distributed because artwork redistribution has not been approved.
`VALIDATION_RESULTS.json` records the automated isolation and source checks.

Still to verify before describing this as a fully reproduced manuscript release:

- Clean installation on a new machine/environment.
- Execution of main Figures 2–6 and Supplementary Figures 4 and 6.
- Visual/numerical comparison with final submitted figures and any manual edits.
- Exact cohort, exclusions, processing provenance, cache provenance and stable
  access to the external input release.
- Full runtimes, peak resource needs and reproducibility across repeated runs.

The existing mammalian input folder contains 226 `.connections.npy` files.
This file count is not a verified analyzed sample size: source functions may
exclude or silently skip subjects, and some analyses sample brains/nodes.
The runner seeds Python/NumPy, but original manuscript seeds were not recorded
here. Existing cached projections and parallel operations can affect results.

The existing Figure 6 tract cache is approximately 1.44 GB and is kept external.
Loading it can require substantial memory; it was not loaded during validation.

## Revision analyses — 17 November 2026

Five commands added for the revised Supplementary Information (`supfigure_distance_metric`,
`supfigure_min_instances`, `supfigure_node_profiles`, `supfigure_hcp`, `suptable_modules`) were run
through `run_figures.py` in the reference environment, with an external data directory containing the
processed mammalian connectomes, the realization files (`brain_nets_mat/`), the metadata and a cache of
the public HCP connectomes. All five completed with exit code 0 and wrote their figures
(`SuppFigures/S_*.png`) and table (`SuppTables/S_modules.csv`). Every number they print matches the
values reported in Supplementary Sections 3.4–3.5, 4, 5.2, 5.3, 5.5 and 11, including the synthetic
controls (seed 0). `suptable_modules`, the slowest command, took about 29 minutes. The exported files
match their recorded source hashes in `SOURCE_MANIFEST.json`.
