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

## 7. Status and open questions

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
- Voxel-level Monte Carlo FWE by relocating foci within the mask.

Known gaps, roughly in priority order:

1. **`π` is weakly identified**, and that is what drives the residual over-correction. It is
   identified only through the *count* of reporting studies given `µ`; a prior on `π`, or
   borrowing strength spatially (neighbouring voxels have similar prevalence), should sharpen it.
2. **Inference is the weakest part.** Per-voxel standard errors come from the observed
   information with `τ²` and the EM responsibilities held fixed, so they are optimistic. The
   Monte Carlo null is voxel-level only; cluster-level (size/mass) is not implemented.
3. **The reporting threshold is still inferred, not known.** `"pooled-min"` is a bound, not the
   truth, and the fit is sensitive to it. Real thresholds are usually stated in the paper and
   should become a first-class metadata field.
4. **Peak-height bias is not corrected.** We model the selection event as `|ĝ| > c` but treat
   the reported value as an unbiased draw. It is really the height of a *local maximum*, which
   is further inflated. The overshoot distribution is the fix.
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
