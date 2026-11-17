# External inputs

Data, cached results, and artwork are deliberately not distributed in this
repository. The manuscript identifies the source study as Assaf et al. (2020),
"Conservation of brain connectivity and wiring across the mammalian class".
The current manuscript states that source data are available on request from
that study's corresponding authors. The authors still need to confirm access
to the exact processed cohort, five human connectomes, derived inputs and
caches used here. No public download URL for that exact release is asserted.

Request the processed inputs and their provenance from the manuscript authors.
Original source data alone may not contain the processing and caches expected
by these scripts. Place approved inputs here locally, or pass `--data-dir` to
the runner. Files placed here are ignored by Git.

Expected layout (paths are relative to the selected data directory):

```text
brain_nets_fixed_means_knn/
    <brain>.connections.npy
    <brain>.distances.npy
    <brain>.fibers.npy [optional; see below]
    <brain>.positions.npy
brain_nets_raw/
    <brain>Surface.nii OR <brain>_1vol.nii
brain_nets_mat/
    <brain>_FinalFinal2.mat [the 20 tractography realizations; field conmatT]
hcp_atlas/
    [created automatically by supfigure_hcp; public HCP connectomes from Open NeuroData]
brain_net_control/
    results.mat OR <brain>.npy control ensembles
brain_nets_tracts/streamlines_core/
    <brain>_core.mat
human_brains/
    processed/<brain>.<connections|distances|fibers|positions>.npy
    mri/<brain>Surface.nii OR <brain>_1vol.nii
    core_tracts_cleaned/<brain>_core.mat
MamiNum_formatted.csv
images/
    animals/<animal>.png
    [figure-specific illustrations]
    figure6_pickle/nodes_cache.pkl
    figure6_pickle/tracts_cache.pkl
    figure6_pickle/data_cache.pkl
```

Connectome adjacency, distance and fiber arrays are N-by-N and positions are
N-by-3. Preserve original brain names, array ordering and units; loaders perform
normalization. When `.fibers.npy` is absent, `BrainNet` derives fiber weights
from distance times adjacency; this is the existing loader behavior and is used
by the local mammalian dataset. Do not substitute a different reconstruction
without recording it. The available files determine `BrainNet.options()`, so an
incomplete cohort changes results. Metadata must include the fields expected by
`load_orders()` (`brain_name`, `Order`, `AnimalName`). MRI volumes and tract
coordinates must match the processed connectomes.

Input groups by figure:

| Figures | Required groups |
| --- | --- |
| All | Processed mammalian connectomes |
| 1 | Animal silhouettes; `figure1_illustration.png`, `figure1_rho_explained.png` |
| 2 | Mammalian MRI; silhouettes; `figure_2_core_illustration.png`, `figure_2_periphery_illustration.png` |
| 3 | Animal silhouettes |
| 4 | Animal silhouettes; `figure3_illustration.png`, `figure4_path_illustraion.png`, `figure4_illustration.png`, `figure4_3d_illustration.png`, `figure4_2d_illustration.png` |
| 5 | Mammalian MRI; silhouettes; control ensembles or `results.mat`; `full-brain-illustration.png`, `hemi-brain-illustration.png`, `quad-brain-illustration.png`, `octo-brain-illustration.png`, `figure5_illustration_cuts_control.png`, `figure5_illustration_cuts_actual.png` |
| 6 | MRI, tracts, human connectomes, metadata, silhouettes, three pickle caches; `brain_projection_illustration_a.png`, `_b.png`, `_c.png` |
| Supplementary 2 and 4 | Processed mammalian connectomes |
| Supplementary 6 | MRI, tracts, human connectomes, metadata and caches (via Figure 6 import) |
| Distance metric (SI 4, 5.3) | Processed mammalian connectomes; metadata (`Order`) |
| Link thresholds (SI 3.5) | Processed mammalian connectomes; `brain_nets_mat/` realization files |
| Node profiles (SI 5.2), modules (SI 5.5) | Processed mammalian connectomes |
| HCP (SI 11) | None locally: Schaefer-200/400 connectomes of the HCP-1200 release are downloaded from `s3://open-neurodata/hcp1200`, atlases from `s3://open-neurodata/neuroparc` |

Figure 5 can write `brain_net_control/results.mat` when absent. Figure 6 caches
are currently mandatory at import; cached projections may be recomputed in
memory for missing keys. Cache-free reproduction and cache provenance have not
been validated. Do not load pickle files or object-containing NumPy files from
untrusted sources.

Some silhouette source references in `Code/style.py` point to Shutterstock.
Neither those images nor other illustration files are included. Obtain the
necessary rights or replace them with approved assets; a code license would
not confer image or data rights. The scripts use local animal PNGs and do not
need to download the linked images.
