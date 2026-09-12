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

## 10. Status and open questions

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
