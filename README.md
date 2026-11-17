# A two-tier architecture of the mammalian connectome

Publication code for the manuscript of this title. This repository contains the
main and supplementary figure scripts and their supporting analysis modules.
It starts from a curated snapshot; private development history, manuscript
drafts, exploratory notebooks, data, and artwork are excluded.

Public repository: [JudahGazit/two-tiers-mammalian-brain](https://github.com/JudahGazit/two-tiers-mammalian-brain).

**Release status:** prepared for author review. Full manuscript reproduction is
not yet certified. Data and artwork must be obtained separately, and a code
license has not yet been selected. See [VALIDATION.md](VALIDATION.md).

## Installation

The reference environment is Windows with CPython 3.13.6. Other platforms have
not been validated. No GPU is explicitly required by the included scripts;
whole-cohort analyses and MRI/tract processing can require substantial RAM and
CPU time. Calibri was used for the figures; Matplotlib substitutes another font
if it is unavailable, which can change layout. Do not redistribute font files.

From this repository's root:

```text
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python run_figures.py --demo
```

On macOS/Linux use `.venv/bin/python` instead. `requirements.txt` pins direct
dependencies; `requirements-lock.txt` records their installed dependency closure
for the reference environment. A clean installation must still be validated.
Installation and full-cohort runtimes have not yet been measured.

The demonstration needs no external data or artwork. It creates
`outputs/demo.png` and `outputs/demo.json`, checking routing against NetworkX and
an identity brain projection numerically. It is not a manuscript result.

## Reproducing figures

Obtain the inputs described in [Datasets/README.md](Datasets/README.md). Supply
the directory containing the dataset folders, not an individual brain directory:

```text
python run_figures.py all --check --data-dir "PATH/TO/Datasets"
python run_figures.py figure1 --data-dir "PATH/TO/Datasets"
python run_figures.py figure2 figure3 --data-dir "PATH/TO/Datasets"
python run_figures.py all --data-dir "PATH/TO/Datasets" --seed 0
```

Use the Python executable from the environment above. Commands work regardless
of the current working directory. `--output-dir` selects an alternative output
directory. `--check` checks installed direct dependency versions and coarse input
presence; it does not establish complete subject coverage or scientific validity.
Each figure runs in a separate process. The default seed is 0 for Python and
NumPy; this was added for repeatability and is not asserted to be the seed used
for the submitted artwork. Parallel operations and existing caches can still
affect repeatability. Execution timings and return codes go to
`outputs/run-report.json`.

| Command | Script | Output under `outputs/` |
| --- | --- | --- |
| `figure1` through `figure5` | `Code/figures/figureN.py` | `Figures/figureN.pdf` |
| `figure6` | `Code/figures/figure6.py` | `Figures/figure6.pdf` and `.png` |
| `supfigure2` | `Code/figures/supfigure2.py` | `SuppFigures/figure2_3.png` |
| `supfigure4` | `Code/figures/supfigure4.py` | `SuppFigures/figure4_0.png` |
| `supfigure6` | `Code/figures/supfigure6.py` | `SuppFigures/figure6_0.png` through `_2.png` |
| `supfigure_distance_metric` | `Code/figures/supfigure_distance_metric.py` | `SuppFigures/S_euclid_orders.png`, `S_euclid_shape.png`, `S_euclid_ev.png` |
| `supfigure_min_instances` | `Code/figures/supfigure_min_instances.py` | `SuppFigures/S_min_instances.png` |
| `supfigure_node_profiles` | `Code/figures/supfigure_node_profiles.py` | `SuppFigures/S_node_profiles.png` |
| `supfigure_hcp` | `Code/figures/supfigure_hcp.py` | `SuppFigures/S_hcp.png` |
| `suptable_modules` | `Code/figures/suptable_modules.py` | `SuppTables/S_modules.csv` (table printed to the console) |

The last five commands produce the supplementary analyses added in the revision (Supplementary
Sections 3.4, 3.5, 4, 5.2, 5.3, 5.5 and 11). They are built on `Code/analysis/`, and print the numbers
quoted in the Supplementary Information. `supfigure_hcp` downloads 200 public atlas connectomes
of the Human Connectome Project from Open NeuroData on first use and caches them under
`<data-dir>/hcp_atlas/`; it requires internet access.

The scripts determine the active supplementary panels. Several alternate
Supplementary Figure 2 functions exist but are disabled in the source; this
package preserves that selection. Authors must confirm correspondence to the
final manuscript figures, including any manual composition or postprocessing.

## Code organization and provenance

`Code/analysis/` holds the revision analyses (tail test, Euclidean and brain-shape
controls, link-threshold control, node-level linking profiles and their synthetic controls,
modules, and the HCP connectomes). `Code/dataloaders/` loads processed connectomes and includes interpolation
helpers; `Code/core_periphery/` defines the separation; `Code/metrics/` computes
wiring and routing measures; `Code/projections/` aligns and projects brains and
tracts. `Code/figures/` assembles the figures. `SOURCE_MANIFEST.json` records
source and exported hashes and the source revision. Portability changes resolve
input/output paths and remove a machine-specific FFmpeg location; exported
whitespace and line endings are normalized. Scientific
calculations were retained. FFmpeg is optional for animation helpers and is not
needed for the static figure commands.

Some analyses skip failed subjects using broad exception handlers. Figure 6
loads external pickle caches on import; use only trusted author-provided files.
The exact cohort, exclusions, cache provenance and correspondence of these
processed inputs to the paper require author verification. Processed data are
the entry point; this package is not a complete raw MRI reconstruction pipeline.

## License and citation

The authors have not selected a code license. No open-source license is granted
by this snapshot. Data and artwork have separate access and usage conditions.
Add the approved license before describing the software as open source.

The manuscript citation, release identifier and archival DOI will be added when
available. Cite the archived manuscript release once it
is deposited. See [RELEASE.md](RELEASE.md) for the publication steps.
