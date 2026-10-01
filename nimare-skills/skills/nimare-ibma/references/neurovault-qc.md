# Selecting NeuroVault images for IBMA

NeuroVault is a voluntary repository: ~238,000 maps in ~5,756 collections as of early
2024, of which roughly 1,473 collections link to a publication. Annotation quality is
uneven. For IBMA, **selection is the analysis** — document every filter.

## Stage 1: metadata filters

Applied to the NeuroVault metadata tables before downloading anything.

- `modality` / `map_type`: fMRI-BOLD, and **T or Z statistic maps only**. Exclude
  parametric maps that are not test statistics.
- `analysis_level`: group (not single-subject).
- `number_of_subjects` > 10.
- `is_thresholded` false. Thresholded maps break every IBMA estimator's assumptions.
- `target_template_image` in MNI space.
- Brain coverage > ~40% of the template mask. Partial-coverage maps otherwise silently
  shrink the shared voxel set.
- A resolvable link to a publication (DOI, or a PubMed match on the collection title).

## Stage 2: heuristic cleaning after download

- **Extreme values.** Z maps whose absolute maximum falls outside roughly 1.96-50 are
  usually not Z maps. Drop them.
- **Duplicates.** The same map uploaded in several collections. Compare on voxel data,
  not filename.
- **Inverted contrasts.** The same contrast uploaded in both directions; keep one and
  record the sign convention.
- **Non-statistical maps by name.** Filenames containing `ICA`, `PCA`, `correlation`,
  `mask`, `ROI`, `atlas` are rarely group statistic maps.
- **Resolution and affine.** Resample to one grid. NiMARE's masker resamples on use, but
  do it explicitly so you can see what happened.

## Stage 3: build the study set

```python
from nimare.generate import create_neurovault_studyset

studyset = create_neurovault_studyset(
    collection_ids={"smith2019": 1234, "jones2020": 5678},
    contrasts={"faces_gt_houses": "as-Face"},   # regex over NeuroVault image names
    img_dir="data/images/",
    map_type_conversion={"T map": "t", "Z map": "z"},
)
```

Then convert to the estimator's required inputs:

```python
from nimare.transforms import ImageTransformer
studyset = ImageTransformer(target=["z", "varcope"]).transform(studyset)
```

Converting t to z requires the right degrees of freedom — supply `sample_sizes` metadata
and let `nimare.transforms.sample_sizes_to_dof` handle it rather than assuming a value.

## Stage 4: dependence

One collection often holds several contrasts from one sample. `groupby=None` groups by
`study_id`, so name your collections by **sample**, not by paper, when a paper has more
than one. See the dependence section of the IBMA skill.

## Stage 5: run and diagnose

```python
from nimare.workflows import IBMAWorkflow
cres = IBMAWorkflow(estimator="dersimonianlaird", corrector="fdr",
                    diagnostics="jackknife").fit(studyset)
```

Jackknife over studies is especially important here: NeuroVault samples are opportunistic,
not a systematic review, so a result carried by one collection is common.

## Reporting

Report the full filter chain with counts at each stage (a PRISMA-style flow works well),
the final number of maps and of independent samples, the statistic type, the
transformations applied, and the estimator with its dependence settings.
