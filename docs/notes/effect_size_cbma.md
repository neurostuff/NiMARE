# Coordinate-based effect-size meta-analysis: landscape, first principles, and the CBES design

Status: design note for the prototype in `nimare/meta/cbma/effectsize.py`. Not yet validated
against a reference implementation; see **Open questions** before using it on real data.

## 1. The question

Every coordinate-based estimator NiMARE ships answers a question about *convergence*: ALE and
MKDA ask whether studies report peaks in the same place more often than chance; CBMR asks how
the *intensity* of reported foci varies over the brain and with study-level covariates. None of
them answers the question a reader of a meta-analysis actually asks — **how big is the effect
there?** Sample size enters ALE only as kernel width and CBMR only as a regressor on foci
counts; the reported `t`/`z` value is discarded entirely.

The goal here is an estimator that consumes `(x, y, z, statistic, N)` and returns a map of
standardized effect size, without imputing effect-size images.

## 2. What already exists

| Method | What it models | Uses reported statistic? | Imputation? |
|---|---|---|---|
| ALE, MKDA, KDA | Convergence of foci | No | No |
| CBMR (Jiang et al., 2024) | Intensity of a foci point process; spline spatial basis; study covariates | No — explicitly listed as a limitation | No |
| SDM / AES-SDM / **SDM-PSI** (Radua, Albajes-Eizagirre) | Effect-size images | Yes | **Yes** — MetaNSUE multiple imputation within bounds, then Rubin's rules |
| **CBRES / ClusterZ** (Tench et al., 2017) | Cluster-wise random-effects meta-analysis of standardized `Z`, with censoring | Yes | **No** — censoring, not imputation |
| BHICP, HPGRF, SBLFR, LGCP regression (Kang, Montagna, Samartsidis, Johnson, Nichols) | Bayesian spatial point processes | No | No |
| `neuropower` (Durnez et al.) | Mixture of null and alternative **peak heights** in one image | Yes | No |

Two conclusions. First, the thing being asked for exists and is respectable: Tench's
CBRES/ClusterZ is a coordinate-based random-effect-size meta-analysis that handles unreported
results by **censoring** rather than imputation, which is exactly the constraint we were given.
Second, nothing in NiMARE does it, and CBRES itself is cluster-wise (it first hard-assigns
coordinates to clusters, then runs a standard meta-analysis per cluster), which throws away
spatial resolution and makes results depend on a clustering step.

So the opportunity is a *voxel-wise*, spatially-modelled version of the CBRES idea.

## 3. First principles: what is a reported peak?

A reported peak is not a random sample of the effect-size field. It is the value at a **local
maximum** of a smooth random field, **conditional on exceeding a within-study threshold**, in a
study that chose to report it. Three distinct distortions follow, and they pull in different
directions:

1. **Winner's curse / selection.** Conditional on `|ẑ| > c`, `ẑ` is biased away from zero. This
   is the same bias GWAS calls winner's curse, astronomy calls Eddington bias/flux boosting, and
   econometrics handles with truncated regression. Its size depends on power, so it is *worst in
   exactly the small studies a meta-analysis most wants to pool*. Our simulations put it at
   +0.39 on a true `g` of 0.3 — larger than the effect itself.
2. **Censoring / non-reporting.** A study that reports nothing at a voxel is informative: its
   local effect probably failed to clear its threshold. Discarding those studies is what makes
   the naive estimate an average over *only the studies that found something*. This is the
   "unreported" information SDM-PSI recovers by imputing; the non-imputation route is a censored
   (Tobit) likelihood term, `P(|ĝ| < c | µ)`.
3. **Peak vs. field.** The reported value is the *maximum* over a neighbourhood, not the value
   at a nominated voxel, and its location carries error (template, registration, smoothing). The
   exact peak-height distribution for a smooth Gaussian field is known (Cheng & Schwartzman;
   Chumbley & Friston's `exp(-u(z-u))` overshoot approximation is the usable form). We currently
   handle only the *location* half of this, via the kernel; see **Open questions**.

### The correction that does *not* work

The obvious fix — treat non-reporting as censoring and fit a plain Tobit model — over-corrects,
and our simulations show it clearly (below). The reason is a modelling error, not a numerical
one: a plain Tobit assumes every study shares one effect `µ` at that voxel, so a silent study
can only be explained as evidence that `µ` is *small*. In reality a silent study usually has **no
effect there at all**. Forcing genuine zeros into a common-effect model drags `µ` toward zero.

## 4. The design: a local zero-inflated Tobit

Borrowing across fields, in the order the ideas do work:

- **Marked point processes** (spatial statistics; `spatstat::Smooth.ppp`). Coordinates with
  attached statistics *are* a marked point pattern. The standard estimator of "the average mark
  near here" is the Nadaraya–Watson smoother — a kernel-weighted mean of the marks. This gives
  the spatial model for free, and it is the same Gaussian kernel ALE already uses, so the
  spatial assumptions stay comparable across estimators.
- **Local likelihood** (Tibshirani & Hastie) / **geographically weighted regression**. Rather
  than smoothing the estimates, use the kernel as *weights in a likelihood* fitted separately at
  each voxel. With all weights equal to 1 it reduces exactly to a textbook random-effects
  meta-analysis, so the spatial part is a weighting scheme, not a new algorithm.
- **Tobit / censored regression** (econometrics), as in CBRES. Silence contributes
  `log P(|ĝ_k| < c_k | µ)`.
- **Zero-inflation / Heckman-style two-part models** (econometrics). Split "does this study have
  an effect here at all" from "how big is it". This is the fix for the over-correction above.
- **Empirical-Bayes deboosting** (astronomy: Jauncey's use of source counts as a prior on true
  flux) and **conditional-likelihood winner's-curse correction** (GWAS) are the cheaper
  alternatives we did *not* take; noted in the roadmap as fallbacks.
- **Preferential sampling** (Diggle, Menezes & Su, 2010) is the fully general framing: the
  locations are sampled preferentially with respect to the field being measured, so location and
  value must be modelled jointly. That is the long-term target (§6), and CBES is a tractable
  approximation to it.

### The model

For study `k` at voxel `v`, with reporting threshold `c_k` on the effect-size scale:

- with probability `π(v)`, the study has a real effect `δ_k ~ N(µ(v), τ²(v))`;
- with probability `1 − π(v)`, it has **no** effect there (`δ_k = 0` exactly);
- it reports a peak iff `|ĝ_k| > c_k`, where `ĝ_k ~ N(δ_k, s_k²)` and `s_k²` comes from `N_k`.

The per-voxel log-likelihood weights each study's *value* contribution by the spatial kernel
`w_k(v)` and each study's *silence* contribution by 1, and is maximized by EM: a closed-form
update for `π` and a one-dimensional Newton step for `µ` (the censored normal likelihood is
log-concave), with `τ²` held at a kernel-weighted DerSimonian–Laird moment estimate.

### Three things that had to be got right

Each of these was found by simulation, and each was worth more than the choice of estimator.

**Two radii, deliberately.** The kernel that weights a study's *value* at `v` is narrow (FWHM
~10–12 mm: how much does a peak 6 mm away tell me about this voxel). The radius that decides
whether a study was *silent* at `v` is wide (default 2× FWHM: did this study report anything in
this region at all). Using one radius for both makes a study with a nearby peak argue against its
own reported effect.

**Silence must not outvote evidence.** A reporting study's log-likelihood is discounted by the
kernel (`w < 1`), so a silent study entering at full weight counts for *more* than a study that
actually measured something. Left alone this dominated everything: the estimate sat ~0.15 below
the truth no matter how many studies reported. Silent studies are now weighted like an average
reporting study at that voxel.

**The reporting threshold has to be pooled, not per-study.** `threshold` defaults to
`"pooled-min"`, the smallest `|z|` reported anywhere. The obvious per-study minimum is badly
biased upward: a study that reported one peak has *no* information about its own threshold, and
its single reported value gets used as one. That inflated threshold makes silence unsurprising,
and the correction quietly does nothing.

**Studies that reported nothing at all must be recovered from the collection.** A study with no
coordinates never reaches `inputs_` — `_collect_inputs` drops it as invalid, which is
neurostuff/NiMARE#294 — yet it is the single most informative observation about `π`. `CBES` takes
its roster of studies from `dataset.ids` and only the reported values from the coordinates table.
Getting this wrong silently removed 18 of 30 studies in testing and pinned `π` at 1.

### What comes out

This is the part worth arguing about, because it reframes the original question. The model
returns **two maps, not one**:

- `prevalence` — `π(v)`, the fraction of studies with a non-null effect here. *This is what
  convergence-based CBMA has been implicitly estimating all along.*
- `g` — `µ(v)`, the effect size **among the studies that have an effect**.
- `g_marginal` — `π · µ`, the population-average effect over all studies, which is what a
  standard random-effects meta-analysis targets.

So convergence and effect size are not competing answers; they are two parameters of one
likelihood, and a method that estimates only one of them is under-specified. That seems like the
most defensible response to "should this be separate from, or an extension of, convergence?"

## 5. Why not extend ALE / MKDA / CBMR instead?

- **ALE / MKDA cannot be extended.** Their statistic is a function of kernel overlap only, and
  their null is "coordinates fall at random". There is no parameter in either model that an
  effect size could identify. Weighting the MA maps by `g` produces a number, but not one that
  estimates anything.
- **CBMR could be, and is the right long-term home.** CBMR already fits a spline-parameterized
  intensity for a point process. The principled extension is a **marked** point process:
  intensity `λ(v)` for where foci are reported, plus a mark distribution for the effect size
  attached to each, sharing a latent field. That is precisely Diggle's preferential-sampling
  model, and it would give a spatially smooth, globally-fitted alternative to CBES's local
  likelihood — with correctly propagated uncertainty, which CBES's per-voxel fits do not have.
  It is a much larger piece of work (the mark model has to carry the threshold `c_k`, and the
  GLM machinery is in torch), so CBES is proposed first as the tractable version that can be
  validated against it later.

## 6. Does it work?

Simulated recovery of a known `g` at the focus, 30 studies, `N ∈ [20, 40]`, `τ = 0.1`, threshold
`p < .001`, 8 seeds per cell. `prev` is the fraction of studies that genuinely have the effect.
Bias in parentheses.

| true `g` | prev | reporting | `none` | `tobit` | `zero-inflated` | est. `π` |
|---|---|---|---|---|---|---|
| 0.3 | 1.0 | 6.4/30 | 0.339 (+0.039) | 0.232 (−0.068) | 0.229 (−0.071) | 0.73 |
| 0.3 | 0.5 | 7.1/30 | 0.272 (−0.028) | 0.122 (−0.178) | **0.162 (−0.138)** | 0.75 |
| 0.5 | 1.0 | 11.5/30 | 0.644 (+0.144) | 0.456 (−0.044) | **0.467 (−0.033)** | 0.95 |
| 0.5 | 0.5 | 7.5/30 | 0.593 (+0.093) | 0.374 (−0.126) | **0.414 (−0.086)** | 0.81 |
| 0.8 | 1.0 | 22.4/30 | 0.788 (−0.012) | 0.698 (−0.102) | 0.704 (−0.096) | 0.99 |
| 0.8 | 0.5 | 13.8/30 | 0.841 (+0.041) | 0.626 (−0.174) | **0.717 (−0.083)** | 0.81 |

Read this honestly:

- **The winner's curse is real and large where it matters.** At `g = 0.5` with a third of studies
  reporting, pooling reported peaks alone overstates the effect by 29%.
- **The correction removes most of it**, and is close to unbiased in the best-identified cell
  (`g = 0.5`, `prev = 1.0`: −0.033).
- **Zero-inflation beats a plain Tobit exactly where predicted** — whenever some studies genuinely
  have no effect, which is the realistic case. The gap is largest at `g = 0.8, prev = 0.5`
  (−0.083 vs −0.174).
- **It still over-corrects by 0.03–0.14**, worst at low prevalence and low effect size. Some of
  this is that `π` is only weakly identified (true 0.5 is estimated at 0.75–0.81), so the zero
  component absorbs less than it should. This is the main thing to fix next.
- **`g` is not a detection map.** A voxel reached by one noise focus has a large `g` and no
  precision behind it. Threshold on `z` (or on `n_studies`), not on `g`.

## 7. Does it recover what the images say?

The strongest available test, because it has a real ground truth. Take the 21 NIDM pain studies,
which have full `t` images. Build a reference by pooling per-study Hedges' `g` across all 21
images voxelwise (random effects, DerSimonian–Laird). Then throw the images away: threshold them
at `p < .001`, keep only the peak coordinates and their `z` values — 2,725 foci, about 1.2% of
voxels — and run CBES on that. Whatever it recovers, it recovers from ~1% of the data.

Reproduce with `python docs/notes/validate_cbes.py images`.

| estimator | r | rho | calibration slope | mean where reference > 0.2 |
|---|---|---|---|---|
| CBES, `none` | 0.735 | 0.770 | 0.21 | 1.056 |
| CBES, `tobit` | 0.797 | 0.839 | 0.34 | 0.798 |
| CBES, `zero-inflated` | 0.780 | 0.822 | 0.31 | 0.848 |
| CBES, `zero-inflated` (`g_marginal`) | **0.802** | **0.840** | 0.35 | 0.792 |
| ALE (`z`) | 0.198 | 0.192 | 0.07 | 0.656 |
| MKDADensity (`z`) | 0.243 | 0.213 | 0.10 | 0.573 |
| *reference* | | | | *0.412* |

A calibration slope is the regression of the reference on the estimate: 1.0 would be perfectly
calibrated, and anything below it means the estimate moves further than the truth does.

Two conclusions, and they point opposite ways.

**Ranking is good, and much better than convergence.** rho = 0.84 against the full-image effect
size, from 1.2% of the data. ALE on the identical coordinates gets 0.19 — which is not a knock on
ALE, it is measuring convergence and is being scored here on a quantity it never claimed to
estimate. It does say that if you want a map that tracks effect magnitude, coordinate density is
close to useless for it and the reported statistics carry almost all of the signal.

**Calibration is poor: the magnitude is roughly twice the truth.** 0.80 against a reference of
0.41. The selection models help (1.06 → 0.80) but do not close it. The reason is gap 4 below and
it is not a tuning problem: a reported peak is a *local maximum*, and its height is inflated
relative to the field around it by an amount the current model does not touch. Correcting for the
*threshold* is not the same as correcting for *being a maximum*, and on real data with N = 9–32
the latter is now the dominant bias. **Treat `g` as a relative map until this is fixed.**

## 8. False positive control

Measured on a global null: 30 studies, 8 noise foci each at uniform random locations with null
peak heights, no effect anywhere. A valid estimator flags ~5% of voxels at uncorrected `p < .05`,
and produces *any* surviving voxel in <=5% of whole simulations after correction.

Reproduce with `python docs/notes/validate_cbes.py fpr 20 100`. 20 simulations, so each
rejection rate has a resolution of 0.05.

| selection model | null | `p<.05` | `p<.01` | bonf | FDR | vFWE | cFWE size | cFWE mass |
|---|---|---|---|---|---|---|---|---|
| `none` | parametric | **0.407** | **0.394** | **1.00** | **1.00** | — | — | — |
| `none` | montecarlo | 0.052 | 0.013 | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| `zero-inflated` | parametric | **0.106** | **0.051** | **1.00** | **1.00** | — | — | — |
| `zero-inflated` | montecarlo | 0.042 | 0.009 | 0.10 | 0.10 | 0.10 | 0.05 | 0.05 |

The first two columns should read 0.05 and 0.01. The parametric rows do not, and the failure is
not marginal: `selection_model="none"` calls 40% of the brain significant at `p < .05` when
nothing is there.

`g / se` is simply not a null-referenced statistic. The standard error treats `tau2` and the
mixture weights as known, and — more fundamentally — every peak being pooled was selected for
being large, so under the null the pooled effect at a focus is large and "significant" by
construction. The zero-inflated model helps a lot (0.40 → 0.10), because widespread silence
pushes `π` and `µ` down, but it does not fix the reference distribution. Bonferroni and FDR
inherit the problem and reject in *every* null simulation. They are not broken; they are
faithfully correcting p-values that were already meaningless.

The fix is the one ALE and MKDA already use: get the uncorrected p-values from a **spatial null**
rather than from a standard error. `null_method="montecarlo"` (the default) relocates every focus
to a random in-mask voxel, keeping its effect size and study membership, refits, and reads `p` off
the resulting distribution of `|z|`. That restores calibration — 0.052 and 0.042 against a
nominal 0.05, 0.013 and 0.009 against a nominal 0.01 — and with it every correction built on
those p-values.

Read the corrected columns with the sample size in mind: 20 simulations resolves a rejection
rate only to 0.05, so every cell above is either 1/20 or 2/20. The three cells at 0.10
(Bonferroni, FDR and voxel FWE under the zero-inflated model) are 2 of 20, which a true rate of
0.05 produces 26% of the time — so this run is consistent with correct control but cannot
demonstrate it. Nothing here rules out a mild inflation at the voxel level under the
zero-inflated model, and confirming it either way needs a few hundred simulations. The cluster
levels came in at 1/20 in every configuration.

Two caveats on the null. It tests the same hypothesis the convergence estimators test — that
reported coordinates fall at random within the mask — so a significant voxel means "more
effect here than random placement of these same reported effects would give", not "the effect
is non-zero". And pooling every voxel of every iteration into one histogram assumes voxels
share a null distribution, which is only approximately true because coverage varies; this is
what ALE and MKDA do too.

**Which correction to use.** Four are available, and all four are valid on the Monte Carlo null:

| correction | call | notes |
|---|---|---|
| Voxel FWE | `FWECorrector(method="montecarlo")` | `logp_level-voxel`. Free once fitted — reuses the null from `fit`. |
| **Cluster FWE (size)** | same | `logp_desc-size_level-cluster`. The usual choice in neuroimaging. |
| **Cluster FWE (mass)** | same | `logp_desc-mass_level-cluster`. More sensitive to tall clusters than to broad ones. |
| FDR | `FDRCorrector()` | Operates on the uncorrected `p` map. |
| Bonferroni | `FWECorrector(method="bonferroni")` | Likewise. Conservative but exact. |

All of them are meaningless with `null_method="parametric"`, since every one of them is a
function of p-values that method does not calibrate.

Clusters are formed on `|z|` at the statistic corresponding to `voxel_thresh` **read off the
null**, not assumed: CBES's `z` is not standard normal, so a nominal 3.29 is not a p of .001.
That threshold is only knowable after permuting, which would ordinarily mean a second pass over
the permutations. Instead a short pilot run (`_NULL_PILOT_ITERS`, or 5% of `n_iters`) fixes the
threshold first, and `fit` then records cluster size and mass alongside the voxel-level null in
the *same* refits. Cluster correction therefore costs about 5% more than voxel correction rather
than 100% more. Pass `cluster_threshold=None` to skip it.

**Cost.** The null dominates, because each iteration is a full refit — unlike ALE, whose
per-iteration statistic is cheap. Measured on the 21-study pain data over 228k voxels, per null
iteration:

| selection model | per iteration | 1000 iterations, 1 core |
|---|---|---|
| `none` | 0.56 s | ~9 min |
| `zero-inflated` | 4.6 s | ~1.3 h |

That is 5.3x faster than the first working version (24 s per iteration, ~6.7 h); see §9. Use
`n_cores` to divide it further, and explore with `null_method="parametric"` before committing to
the inference you intend to report.

## 9. Making the null affordable

A permutation here is a full refit, so the estimator's speed *is* its inference budget. The
first working version took 24 s per whole-brain zero-inflated fit, which put a 1000-iteration
null at 6.7 hours. It is now 4.6 s, and cluster-level correction rides along in the same pass.
What actually mattered, in order:

1. **Iterate over weighted pairs, not the dense block** (4.3x). The EM built
   `(n_studies x n_voxels)` arrays and evaluated normal CDFs across all of them — then multiplied
   most of the results by zero. At any voxel a study has either reported nearby or been silent
   there; most studies are neither, because they reported in the region but outside this voxel's
   kernel. Visiting only the pairs that carry weight left the arithmetic identical and took the
   EM from 97% of runtime to about half of it.
2. **A scratch bitmap instead of `np.unique`** for deduplicating the voxels a study's coverage
   spheres reach. With a 20 mm sphere per focus there are a great many hits to sort.
3. **`scipy.special.ndtr` and an inlined normal density** instead of `scipy.stats.norm`, which
   is about 3x slower on large arrays.
4. **Retiring converged voxels from the working set** (a further 15% on null iterations). Voxels
   converge at wildly different rates — 21 iterations passed before even a quarter of them had
   settled — so iterating the whole block until the slowest one is done wastes most of the work.
   Stopping early on a global criterion instead would leave the stragglers short of the MLE.
5. **A pilot run for the cluster-forming threshold**, so cluster measures are recorded during
   the same permutations as everything else rather than in a second pass (§8).

Two things tried that did *not* pay and were kept only where they earned it: lowering the
compaction threshold below 0.05 (the working set does not shrink fast enough for it to matter),
and active-set shrinking on the *observed* fit (neutral at the default iteration budget — it
earns its place on the null iterations, where fewer studies reach each voxel).

## 10. Mixing images with coordinates, and the peak-height bias

An image study is the limiting case of a coordinate study: it gives the effect at a voxel with
no localization uncertainty and no reporting threshold. So it enters the same likelihood with
kernel weight 1 everywhere and never contributes a censoring term — `use_images=True`, and a
collection may mix the two freely. Nothing about the mixture or the EM changes.

### How big is the peak-height bias really?

Not the 2x reported in §7. That figure compared a reported peak to its *own study's*
neighbourhood, which still contains that study's noise — and the noise is what put the peak
there. Comparing instead against a **leave-one-out** pooling of the other 20 pain studies:

| distance from reported peak | reported \|g\| | true \|g\| (leave-one-out) | ratio |
|---|---|---|---|
| 2 mm | 1.246 | 0.267 | 0.214 |
| 8 mm | 1.216 | 0.241 | 0.198 |
| 20 mm | 1.215 | 0.224 | 0.184 |

A reported peak overstates the true local effect roughly **fivefold**. The selection model
already removes about half of that, leaving the ~2x residual §7 measured against the reference.

### The correction, and why the obvious calibration is wrong

`peak_bias=rho` treats a reported statistic as measuring `g_true / rho` and rescales the effect
by `rho`, its variance by `rho**2` and the study's reporting threshold by `rho` — a rescaling of
the effect-size axis for coordinate studies, which leaves the censored likelihood coherent.
Images are never rescaled. Because it is an exact rescaling, the fitted map scales *exactly*
linearly in `rho`, which is what makes `rho` calibratable from a single ratio.

Two calibrations that do **not** work:

- **The raw peak-to-truth ratio (0.21).** It corrects twice: the selection model has already
  removed part of the bias. Applying it drove the estimate to 0.18 against a reference of 0.41.
- **A regression slope of image map on coordinate map (0.31).** Attenuated by the many voxels
  where one study peaked and the images say nothing. It over-corrected to 0.61x.

What works is the ratio of the summaries being compared, calibrated on studies that supply
*both*: fit them from their images, fit the same studies from their coordinates, take the ratio.
Split-half on the pain collection, applied to held-out studies:

```
rho = 0.479 +- 0.056       coordinate/image ratio: 1.96x before -> 0.94x after
```

### Does it make the estimator coherent?

The test that matters is invariance: an estimator whose answer depends on how many studies
happened to share images is incoherent, whatever its correlation with the truth. Sweeping the
number of studies supplied as images, on the pain collection:

| images used | uncorrected | `peak_bias=0.479` |
|---|---|---|
| 0 | 0.848 | 0.406 |
| 5 | 0.540 | 0.404 |
| 10 | 0.478 | 0.409 |
| 21 | 0.447 | 0.447 |
| **spread** | **0.401** | **0.045** |

A ninefold reduction in provenance dependence, landing on the reference value of 0.412.
Correlation still climbs with more images (0.78 to 0.96), as it should — images carry more
information, they just no longer carry a *different answer*.

### Validated against a known truth

The generator in `nimare.generate` draws one value at the ground-truth location and thresholds
it. That models the reporting threshold but never the *selection of a location*, so it cannot
produce the peak-height bias and could not be used to test a correction for it. A random-field
generator (`docs/notes/validate_cbes.py`) that reports genuine local maxima of a smooth field
does, and there the truth is known exactly:

| | mean g over the true blob | r with truth |
|---|---|---|
| true field | 0.227 | |
| coordinates, uncorrected | 0.583 | 0.039 |
| coordinates, `peak_bias` calibrated on held-out images | **0.191** | 0.039 |
| images | 0.225 | 0.499 |

The level is recovered. **The spatial pattern is not**: r = 0.04 against the truth in this
regime, against 0.50 for images. With 20 studies reporting ~5 peaks each into a 27,000-voxel
volume, most reported peaks are noise, and no rescaling can fix where they are. The pain
collection is far kinder (2,725 peaks, r = 0.78) because real papers report many more foci.

## 11. Where this fails: NeuroVault

The same protocol on NiMARE's default 11-collection NeuroVault studyset does not work, and the
reason is not subtle. Thresholding those images at p < .001 leaves **3 studies with any peak at
all** (35 foci); relaxing to p < .05 reaches only 4 studies. Out-of-sample calibration on such a
collection is degenerate — the splits repeat — and the corrected coordinate/image ratio was
2.03x rather than the 0.94x seen on the pain data.

`rho` is also not a transferable constant. In-sample it ranged from 0.62 to 0.27 across
reporting thresholds on the same NeuroVault studies, against 0.479 on the pain collection. It
depends on the threshold, the smoothness and the study sizes, so **it has to be calibrated on
the collection being analysed**, which requires enough studies supplying images *and*
coordinates. A default value would be worse than none.

## 12. The peak-height correction that needs no images, and why it cannot work

The obvious objection to §10 is that calibrating `peak_bias` needs images, and if you had
images you would not be doing coordinate-based meta-analysis. The principled alternative is to
get the correction from random field theory instead: a reported peak is a local maximum of a
smooth field, and RFT gives its height distribution, so the true effect could in principle be
deconvolved from the reported statistic alone.

That was implemented and it does not work. The reason is worth recording, because it is a fact
about the data rather than about the implementation.

**The machinery is sound.** A mixture of null peaks (RFT density) and signal peaks recovers its
own parameters on synthetic data (`pi0` 0.70/0.50/0.90 recovered as 0.71/0.49/0.90). The null
density checks out against simulated pure-noise fields to within +0.17 z units at fMRI-like
smoothness. Modelling a signal peak correctly — as *noncentrality plus peak overshoot*, since a
local maximum of a smooth field sits above its mean even with no selection, not as
`N(lambda, 1)` — took the simulated bias from 0.81 to 0.21 against a truth of 0.07.

**But there is nothing to deconvolve.** On the 21 NIDM pain studies:

| | z units |
|---|---|
| mean reported peak height | 3.639 |
| mean height of a *pure noise* peak at the same threshold | 3.625 |
| **excess** | **+0.009** |
| what the effect actually present at those locations would give | +1.104 |

The reported peaks are statistically indistinguishable from peaks of pure noise. With N ~ 16 and
a true effect near 0.27, no voxel has an appreciable chance of clearing z = 3.29 on signal, so
the peaks that get reported are wherever the noise happened to be largest, and their heights are
set by the threshold, not by the effect.

**An ablation confirms it.** Replacing every reported statistic with a constant, or with a draw
from the null peak distribution, barely moves the result:

| coordinate statistics | r with image truth |
|---|---|
| reported, as published | 0.780 |
| all set to a constant | 0.754 |
| redrawn from the null | 0.755 |
| ALE (uses no statistics) | 0.198 |

The reported magnitudes are worth 0.026 of correlation. What carries the signal is *where* the
peaks are, and the sample sizes attached to them.

**So the honest conclusion is a negative one.** No correction computed from peak values can
recover the effect size in this regime, because the values do not contain it. The
image-calibrated `peak_bias` of §10 works precisely *because* it does not try: the bias is a
near-deterministic function of the reporting threshold and the sample size, so a single scalar
removes it. That also explains why it is not transferable — it is a property of the collection's
thresholds and study sizes, not of the brain.

`peak_information()` reports this for any collection, using nothing but the reported statistics
and the threshold, and :class:`CBES` warns on `fit` when the excess falls below 0.25 z units.
When it does, read the effect-size map as a relative one: its spatial pattern is still driven by
where the peaks are, which this finding does not touch.

For collections of well-powered studies the excess would be substantial and the deconvolution
would have something to work on. The machinery is in `docs/notes/` rather than the estimator
because nothing on hand could demonstrate it working.

## 13. Practical questions: how many images, and are coordinates worth adding?

### How many studies with images does calibrating `peak_bias` need?

Because the fitted map scales exactly linearly in `rho`, the relative sampling error of `rho`
*is* the relative error of the reported effect sizes. Twelve independent draws per point, on the
pain collection:

| images used to calibrate | mean `rho` | relative SD = error in `g` |
|---|---|---|
| 2 | 0.389 | ±145% |
| 3 | 0.504 | ±26% |
| 4 | 0.461 | ±24% |
| 5 | 0.441 | ±17% |
| 8 | 0.439 | ±12% |
| 12 | 0.470 | ±13% |
| 16 | 0.490 | ±6% |

Below five images the calibration is too noisy to be worth applying; eight gives ±12%, sixteen
±6%. This is also why NeuroVault failed at three (§11) — that sits in the ±26% band, on top of
degenerate splits.

### Reported magnitudes track the reporting threshold, not the brain

Thresholding the same 21 studies at different levels, against what a model with **no effect in
it** predicts from the threshold and sample size alone:

| threshold used | mean reported \|g\| | null-peak prediction |
|---|---|---|
| p < .001 two-tailed (z = 3.29) | 1.259 | 1.233 |
| FWE-ish (z = 4.26) | 1.650 | 1.825 |

Predicted to within 2% and 10% by pure noise. A paper using FWE correction contributes
\|g\| ~ 1.65 where a paper using p < .001 contributes ~1.26, for the same brain.

**Consequences for a mixed-threshold literature**, which is the normal case when pulling from
published papers: a single scalar `rho` is *not* valid, because it assumes a common threshold.
`threshold="pooled-min"` is also wrong there — it applies the most lenient study's threshold to
everyone. Use each paper's stated threshold (the parameter accepts a metadata field name), or
`"study-min"` as a fallback. The correction such a collection actually needs is per-study,
`rho(u_k, N_k)`, which is computable from the paper alone and is not implemented.

### Do coordinate studies add anything on top of images?

Truth is the 21-study image pooling; five random splits.

| configuration | r with truth | mean g (truth 0.412) |
|---|---|---|
| 10 images only | 0.893 | 0.473 |
| 10 images + 11 coordinate studies, uncorrected | 0.903 | 0.490 |
| 10 images + 11 coordinate studies, `peak_bias` | **0.930** | **0.417** |
| 15 images only | 0.932 | 0.469 |
| 21 images only | 0.956 | 0.447 |
| 0 images, 21 coordinate studies | 0.780 | 0.406 |

Yes, modestly: **the exchange rate is about two coordinate studies per image.** Eleven
coordinate papers bought roughly five images' worth (0.930 against 0.932 for 15 images).

But the gain is unlocked by the bias correction. Uncorrected, the coordinates add +0.01 of
correlation and push the mean *away* from the truth (0.473 to 0.490). Corrected, they add +0.037
and land the mean on 0.417 against 0.412.

Given the ablation of §12 — reported magnitudes are worth 0.026 of correlation — the split is
that **the spatial gain is robust** (it comes from where the peaks are and the sample sizes,
neither threshold-dependent) while **the magnitude gain is not**. For a mixed-threshold
collection: add the coordinates for localization and inference, take the scale from the images.

Caveats: the truth here includes the studies added as coordinates, so some movement toward it is
structural — the comparison against "15 images only" is the trustworthy one — and `rho = 0.479`
was calibrated on this collection at one threshold.

## 14. Status and open questions

Implemented and working:

- `nimare.meta.cbma.CBES` — the estimator above, with `selection_model` in
  `{"zero-inflated", "tobit", "none"}`, fitted by EM over voxel blocks.
- `nimare.meta.cbma.effectsize.peak_stat_to_hedges_g` — `t`/`z` + `N` → Hedges' `g` and its
  variance, reusing `transforms.t_to_d` / `d_to_g` so coordinate- and image-based estimates land
  on one scale.
- `nimare.generate.create_effect_size_coordinate_studyset` — simulates the whole reporting
  process (true effect → study draw → sampling draw → threshold), including `prevalence` for
  genuine zeros and null peak heights from the exponential overshoot approximation. Without a
  simulator that models *thresholding*, none of this can be validated.
- Voxel-level and cluster-level (size and mass) Monte Carlo FWE, and a Monte Carlo null for the
  uncorrected p-values, all computed in one pass over the relocations.

Known gaps, roughly in priority order:

1. **Peak-height bias is not corrected, and it is now the largest error.** We model the
   selection event as `|ĝ| > c` but treat the reported value as an unbiased draw from the field.
   It is really the height of a *local maximum*, which is inflated on top of the thresholding.
   This is what leaves the real-data estimates ~2x high (§7). The fix is to model the reported
   value with the peak-height distribution for a smooth Gaussian field — Chumbley & Friston's
   `exp(-u(z-u))` overshoot approximation under the null, and the Cheng–Schwartzman distribution
   more generally — rather than with the plain normal density now used. **This is the top
   priority**; until it lands, `g` is a relative map.
2. **Monte Carlo inference is still the dominant cost**, at ~1.3 h single-core for a whole-brain
   zero-inflated null at the default 1000 iterations, down from 6.7 h (§9). Options not yet
   taken, in rough order of promise: an *approximate* null that simulates local study
   configurations directly instead of refitting the brain (the analogue of ALE's
   `null_method="approximate"`, and potentially a further 100x for the uncorrected p-values);
   warm-starting each permutation from the previous one; accelerating the EM itself, which
   converges linearly and needs tens of iterations.
3. **`π` is weakly identified**, and that is what drives the residual over-correction in
   simulation. It is identified only through the *count* of reporting studies given `µ`; a prior
   on `π`, or borrowing strength spatially (neighbouring voxels have similar prevalence), should
   sharpen it.
4. **The reporting threshold is still inferred, not known.** `"pooled-min"` is a bound, not the
   truth, and the fit is sensitive to it. Real thresholds are usually stated in the paper and
   should become a first-class metadata field.
5. **"Silent" assumes whole-brain coverage.** An ROI study that never examined a voxel is not
   evidence of a null effect there. There is no flag for this today; it is the natural place for
   a proper Heckman selection equation, where reporting probability depends on covariates
   (ROI vs. whole-brain, journal, sample size) and not on the latent value alone.
6. **Data availability is the real-world blocker.** Neurosynth coordinates carry no statistics
   at all; Sleuth/BrainMap files carry none. NIMADS/NeuroStore points *do* have a `values` field
   (`z_stat`, `t_stat`), so the pipeline is there, but the coverage of that field across
   NeuroStore should be measured before promising anything.
7. Two-sample designs assume equal group sizes; `n1`/`n2` should be read when present.

## References

- Tench, Tanasescu, Constantinescu, Auer & Cottam (2017). Coordinate based random effect size
  meta-analysis of neuroimaging studies. *NeuroImage* 153, 293–306. doi:10.1016/j.neuroimage.2017.04.002
- Albajes-Eizagirre, Solanes & Radua (2019). Meta-analysis of non-statistically significant
  unreported effects. *Stat Methods Med Res*. doi:10.1177/0962280218811349
- Jiang, Nichols et al. (2024). Neuroimaging meta-regression for coordinate-based meta-analysis
  data with a spatial model. *Biostatistics* 25(4), 1210. (CBMR)
- Samartsidis, Montagna, Nichols & Johnson (2019). Bayesian log-Gaussian Cox process regression.
  *JRSS-C* 68(1), 217–234.
- Diggle, Menezes & Su (2010). Geostatistical inference under preferential sampling. *JRSS-C*
  59(2), 191–232.
- Cheng & Schwartzman (2015). Distribution of the height of local maxima of Gaussian random
  fields. *Extremes*.
- Durnez, Degryse, Moerkerke, Seurinck, Sochat, Poldrack & Nichols (2016). Power and sample size
  calculations for fMRI studies based on the prevalence of active peaks. bioRxiv 049429.
- Baddeley, Rubak & Turner (2015). *Spatial Point Patterns: Methodology and Applications with R*
  (Nadaraya–Watson mark smoothing, `Smooth.ppp`).
