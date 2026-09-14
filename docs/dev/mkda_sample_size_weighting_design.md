:orphan:

# Sample-size weighting for MKDA

Design for weighting MKDA density results by study sample size, per
Wager et al. (2009), NeuroImage 45(1 Suppl):S210-21, doi:10.1016/j.neuroimage.2008.10.061
(PMC3318962).

## 0. Decisions

Settled by the primary sources and by measurement. The evidence column points at the
section that carries it.

| # | Decision | Evidence |
|---|---|---|
| D1 | **Weight is `delta_c * sqrt(N_c)`, renormalised across the contrasts in the analysis.** This is Wager 2009's equation verbatim, and it is what the CANlab MATLAB reference computes. | §1.1, §1.2 |
| D2 | **Normalise to sum to `k`, not to 1.** CANlab's weights sum to 1, making the statistic a proportion. NiMARE's MKDA statistic is already `k x` that proportion. Keeping the `k` scaling makes the unweighted path bit-identical and the weighted statistic monotone-equivalent to CANlab's. | §2.1, §3.1 |
| D3 | **Weighting is opt-in via a `weighting=` argument that defaults to `None`.** Turning it on by default would silently change every existing MKDA result. | §3.2 |
| D4 | **The approximate null must become a weighted Poisson-binomial.** Today's exact convolution assumes unit weights. Plugging non-uniform weights into the shipped code and leaving the null alone gives Dice 0.80 against the correct answer at z>1.65, and misses a third of the suprathreshold volume. This is the substantive part of the work, not the weight formula. | §2.2 |
| D5 | **`CBMAEstimator._p_to_summarystat`'s montecarlo branch is broken and must be fixed first.** It compares raw counts to a p-value without normalising or accumulating, so it returns the same threshold for p=0.01 and p=0.001 (6, where the correct answers are 2 and 3). It is currently conservative on integer bins; on the fine grid D4 needs, it flips to returning ~0. | §2.3 |
| D6 | **`N` for a contrast is the *mean* over the analysis's `sample_sizes` entries**, matching what the ALE kernel does with the same field. Every NiMARE converter writes `sample_sizes` as a one-element list per contrast, and the Sleuth exporter reduces it with `min` -- the field is not a per-group decomposition to sum. | §3.3 |
| D7 | **The fixed-effects discount is opt-in and label-driven; never inferred.** NIMADS has no fixed/random field and NiMARE has no such convention. Default `delta = 1` for every contrast; the user names a metadata field to activate `delta = 0.75`. | §1.3, §3.3 |
| D8 | **Missing, zero or NaN `N` imputes the mean weight and warns.** This is CANlab's behaviour. Dropping those studies would silently change the study set relative to the unweighted analysis. | §1.2, §3.5 |
| D9 | **Weights align to MA-map rows by study id, with an assertion — not by positional coincidence.** The current code relies on `groupby("id").first()` happening to match the kernel's `np.unique(id)` ordering. It does today; nothing enforces it. | §2.4 |
| D10 | **`MKDAChi2` and `KDA` are out of scope.** Wager 2009 applies the weights to the density statistic only; its chi-square tests run on unweighted counts. | §1.4, §6 |
| D11 | **Replace the body of `MKDADensity._compute_weights` rather than extend it.** It reads `inference` and `sample_size` columns that `_preprocess_input` never creates for an MKDA kernel, so it has always returned uniform weights. It is dead code with a live-looking `TODO`. | §2.1 |

## 1. Primary sources

### 1.1 Wager et al. (2009), the weighting equation

> "Rather than weighting Z-scores, the current version of MKDA weights by the square root
> of the sample size for each SCM. In addition, we down-weight studies using fixed effects
> analyses by a factor of 0.75."

    P = sum_c I_c ( delta_c sqrt(N_c) / sum_c delta_c sqrt(N_c) )

`c` indexes comparison (contrast) maps, `I_c` is the indicator that the voxel lies within
`r` mm of a peak reported by contrast `c`, `delta_c` is the fixed-effects discount, `N_c`
is the sample size. On the scope of the scheme:

> "This weighting scheme could be used to weight by other study quality measures developed
> by the analyst, such as diagnostic criteria or sample-matching procedures employed in
> studies of psychiatric or medical populations."

Note what `sqrt(N)` is and is not. It is not inverse-variance weighting — that would be
`~N` for a standardised mean difference. It is a weighting on the *evidence* scale: a
study's Z-statistic grows like `sqrt(N) * d`, and `I_c` is a thresholded statistic rather
than an effect size. The choice is a defensible heuristic, not a derived optimum, and the
0.75 discount is arbitrary by the authors' own presentation. §4.1 returns to this.

### 1.2 CANlab reference implementation

`densityUtility3/Meta_Setup.m`, the canonical MATLAB implementation:

```matlab
DB.rootn = sqrt(DB.Subjects);
[DB.connumbers,DB.pointind] = unique(DB.Contrast);
DB.rootn = DB.rootn(DB.pointind);
...
if isfield(DB,'SubjectiveWeights')
    w = DB.rootn .* DB.SubjectiveWeights(DB.pointind);
else
    w = DB.rootn;
end

% make sure no NaNs or zeros; impute mean
whbad = find(isnan(w) | w == 0);
if ~isempty(whbad)
    disp(['Warning! ' num2str(length(whbad)) ' contrasts have bad or missing weights. Imputing mean. ']);
    wok = w; wok(whbad)=[];
    w(whbad) = mean(wok);
end

% these must sum to 1 !
DB.studyweight = w ./ sum(w);
```

Three things follow. The weight is *per contrast*, taken once from the contrast's first
peak. The fixed-effects discount is not automatic — despite what `read_database.m`'s
header says, no shipped CANlab code derives `SubjectiveWeights` from `FixedRandom`; the
analyst supplies it. And bad weights are imputed, not dropped.

`Support_functions_density/meta_prob_activation.m` applies them:

```matlab
for i=1:nc, ivectors(:,i) = ivectors(:,i) .* wts(i); end
activation_proportions = sum(ivectors, 2);
```

`meta_stochastic_activation_blobs.m` — the Monte Carlo null — keeps each weight attached
to its own contrast while the contrast's blobs move (`indic = indic .* MC_Setup.wts(c)`),
and errors outright on NaN weights. `Meta_Select_Contrasts.m` recomputes `w ./ sum(w)`
over the retained subset, which is what D2 and §3.6 have to reproduce for leave-one-out.

For between-condition contrasts, `meta_apply_contrast.m` rescales to mean-1 weights
(`wts = wts ./ mean(wts)`) and runs weighted OLS on the condition indicator matrix, so
the betas are weighted proportions *within* condition. That is the shape a weighted
two-sample MKDA would take; see §6.

### 1.3 Whether the fixed-effects discount is empirically supported

Bossier et al. (2018), "The Influence of Study-Level Inference Models and Study Set Size
on Coordinate-Based fMRI Meta-Analyses" (PMC5778144), concludes that "the best balance
between TPR and FPR detection is observed when using mixed effects group level models
together with a fixed or random effects meta-analysis," and reports fixed-effects
study-level models performing consistently worse in AUC. That supports the *direction* of
a discount. It supports no particular magnitude, and 0.75 remains a convention.

### 1.4 The chi-square tests are unweighted

Wager 2009 describes the specificity analysis on counts: "For the local area around each
voxel, a 'yes/no' by task type contingency table is constructed, where 'yes' and 'no'
refers to whether the SCM activated within r mm of the voxel." The weights do not enter.
Samartsidis et al. (2017), "The coordinate-based meta-analysis of neuroimaging data"
(PMC5849270), states MKDA's statistic as `m(v) = (1/sum_i w_i) sum_i w_i M_i(v)` and notes
"the weights are usually chosen to be proportional to the number of participants" —
again, for the density statistic.

## 2. What NiMARE does today

### 2.1 The weighting hook exists and has never fired

`nimare/meta/cbma/mkda.py:203`:

```python
def _compute_weights(self, ma_values):
    """Determine experiment-wise weights per the conventional MKDA approach."""
    # TODO: Incorporate sample-size and inference metadata extraction and
    # merging into df.
    ids_df = self.inputs_["coordinates"].groupby("id").first()
    n_exp = len(ids_df)
    if "inference" not in ids_df.columns:
        ids_df["inference"] = "rfx"
    if "sample_size" not in ids_df.columns:
        ids_df["sample_size"] = 1.0
    ...
    weight_vec = n_exp * ((np.sqrt(n) * inf) / np.sum(np.sqrt(n) * inf))
```

`CBMAEstimator._preprocess_input` merges `sample_size` into the coordinates frame only
when `"sample_size"` is among the *kernel's* parameters. `MKDAKernel` takes `r` and
`value`, so the column never appears, `n` is always 1.0, `inference` is always `"rfx"`,
and `weight_vec` is always all-ones. The `TODO` describes exactly the gap.

The formula is otherwise right, including the `n_exp` scaling, which is why D2 keeps it:
with unit weights the statistic is the count of activating contrasts, and `k x` CANlab's
proportion. The scaling is monotone, so p- and z-maps are unaffected either way, and
keeping it means the unweighted path does not move.

### 2.2 The approximate null assumes unit weights

`MKDADensity._determine_histogram_bins` sets `histogram_bins = np.arange(k + 1)`, and
`_compute_null_approximate` convolves unit-weight Bernoullis:

```python
ss_hist = 1.0
for exp_prop in prop_active:
    ss_hist = np.convolve(ss_hist, [1 - exp_prop, exp_prop])
```

That is the exact distribution of `sum_c I_c`. It is the wrong distribution for
`sum_c w_c I_c` whenever the weights differ, and nothing in the code notices: the bins
stay integers, the statistic becomes continuous, and the p-values are read off a null
that no longer describes the statistic.

Measured on `nimare/tests/data/nimads_studyset.json` (k=27), comparing the shipped code
with sqrt(N) weights injected against the same weighted statistic scored on a correct
weighted null:

| sample sizes | weight range | ESS | z corr | Dice z>1.65 | Dice z>3.09 |
|---|---|---|---|---|---|
| as in the test set (5-15) | 0.78-1.35 | 26.3/27 | 0.980 | 0.834 | 0.966 |
| resampled to 8-200 | 0.36-1.38 | 24.8/27 | 0.978 | 0.799 | 0.946 |

At z>1.65 the shipped null reports 11,843 voxels where the correct null reports 17,819 —
a 34% shortfall on a corpus whose weights span less than 4x. This is the reason the
feature is more than a one-line weight change.

For comparison, the *effect of weighting itself* on this corpus is mild: with a correct
null throughout, sqrt(N) weighting versus uniform gives z correlation 0.994 and Dice 0.95.
The weighting changes the answer a little; getting the null wrong changes it a lot.

### 2.3 `_p_to_summarystat`'s montecarlo branch

`nimare/meta/cbma/base.py:584`. The approximate branch normalises the histogram and takes
a reverse cumulative sum before searching for `p`. The montecarlo branch does neither:

```python
hist_weights = self.null_distributions_["histweights_corr-none_method-montecarlo"]
ss_idx = np.maximum(0, np.where(hist_weights <= p)[0][0] - 1)
```

For `p < 1`, `hist_weights <= p` means "count is zero", so it returns the bin before the
first never-observed statistic value, independent of `p`. Measured, k=27, 50 iterations:

```
counts: [7435009 3197643 686419 95013 9392 651 23 0 0 ...]
_p_to_summarystat(0.01)  = 6      correct = 2
_p_to_summarystat(0.001) = 6      correct = 3
```

This feeds the cluster-forming threshold in `correct_fwe_montecarlo`
(`base.py:983`) and in ALE (`ale.py:543`), so it is not MKDA-specific. On integer bins it
errs conservative. On the fine grid of §3.4 the first zero-count bin arrives almost
immediately and the threshold collapses toward zero, which would be a silent
false-positive machine. D5 makes fixing it a prerequisite rather than a bundled change,
because it moves unweighted montecarlo results too and deserves its own justification.

### 2.4 Row alignment is a coincidence, not an invariant

`MKDAKernel._transform` orders its output rows by `np.unique(coordinates["id"])`.
`_compute_weights` orders its weights by `groupby("id").first()`. Both sort, so they agree
— verified on the three bundled NIMADS studysets and on a studyset whose ids were
explicitly reversed (`filter_ids` renormalises back to canonical order). Nothing in either
module states the dependency. With uniform weights a misalignment is invisible; with
sample-size weights it silently attaches the wrong N to the wrong contrast. D9.

## 3. Design

### 3.1 The weight

For the contrasts `c = 1..k` in the analysis:

    raw_c   = delta_c * f(N_c)            f = sqrt by default
    w_c     = k * raw_c / sum_c raw_c

`sum_c w_c = k`, so with `delta = 1` and equal `N` every `w_c = 1` and the statistic is
today's count of activating contrasts, unchanged. `w / k` is CANlab's
`Activation_proportion.img`.

### 3.2 Public API

A weighting object, passed to the estimator the way `kernel_transformer` already is:

```python
from nimare.meta.cbma.weights import StudyWeights

MKDADensity(weighting=StudyWeights())                       # sqrt(N), no FFX discount
MKDADensity(weighting="sample_size")                        # string shorthand for the above
MKDADensity(weighting=StudyWeights(inference_field="inference"))
MKDADensity(weighting=StudyWeights(values={"study-1": 2.0, ...}))
MKDADensity()                                               # unchanged, uniform
```

```python
class StudyWeights(NiMAREBase):
    def __init__(
        self,
        source="sample_size",          # "sample_size" | "uniform" | mapping | array
        transform="sqrt",              # "sqrt" | "linear" | "none"
        reduce="sum",                  # collapse per-group sample_sizes: "sum" | "mean"
        inference_field=None,          # metadata field naming the study-level model
        fixed_effects_discount=0.75,
        fixed_effects_labels=("fixed", "ffx", "fixed-effects", "fe"),
        on_missing="impute",           # "impute" | "raise"
    ): ...

    def raw_weights(self, studyset, ids) -> pd.Series:
        """Unnormalised, non-negative, one entry per id, indexed by id."""
```

Why an object rather than five constructor arguments. `MKDADensity.__init__` already
carries seven; adding `weight_by`, `weight_transform`, `weight_reduce`,
`fixed_effects_field` and `fixed_effects_discount` to it would put five parameters on the
estimator that are meaningless whenever the first is `None`. The object keeps the
estimator's surface at one argument, groups parameters that only make sense together,
inherits `get_params`/`set_params` from `NiMAREBase` for free, and — the part that
matters for review — is testable on a studyset and a list of ids with no masker, no
kernel and no imaging.

The seam: **`StudyWeights` produces relative weights; the estimator owns the scale.**
Normalisation cannot live in the weighting object because leave-one-out renormalises over
a subset the object never sees (§3.6, and CANlab's `Meta_Select_Contrasts` does the same).

### 3.3 Where the numbers come from

`sample_sizes` is already a first-class NIMADS metadata field with a normalisation path
(`nimare.studyset.requirements.coerced_sample_sizes`, which accepts `sample_sizes` or
`sample_size`, at analysis or study level). `StudyWeights.raw_weights` reads it through
`_add_metadata_to_dataframe(..., filter_func=np.sum)`.

`reduce="mean"` (D6), matching the ALE kernel's use of the same field. The earlier draft
of this design argued for `"sum"` on the grounds that CANlab's `DB.Subjects` is the number
of subjects behind a contrast and `sample_sizes` might decompose it by group. The codebase
settles it the other way:

- `convert_sleuth_to_dict` writes `metadata["sample_sizes"] = [sample_size]` from
  BrainMap's `//Subjects=` line, which is already the contrast total (`io.py:1375`).
- `convert_neurovault_to_dataset` collects one count per *image* in the contrast, notes
  they "should all be the same", and stores `[modal sample size]` -- summing there would
  multiply `N` by the number of images (`io.py:2036`).
- `_get_sample_size`, NiMARE's Sleuth *exporter*, reduces the list with `min`
  (`io.py:1225`).

Nothing in NiMARE treats the list as summable, and on a NeuroVault-derived corpus `"sum"`
would inflate `N` several-fold. `reduce` stays exposed for an analyst whose metadata means
something else.

`sample_sizes` must **not** be declared in `_required_inputs`. That path
(`requirements.PerAnalysis.validity`) drops analyses that lack the field when
`drop_invalid=True`, which would silently shrink the study set the moment weighting is
switched on. D8 imputes instead.

`inference_field` names a metadata field whose values are matched case-insensitively
against `fixed_effects_labels`; matches get `delta = 0.75`, everything else gets 1.0. No
field, no discount. NiMARE has no fixed/random convention to infer from, and guessing
wrong systematically biases the result, so the analyst has to say it out loud (D7).

### 3.4 The null distribution

The statistic is `S = sum_c w_c I_c`, a weighted sum of independent Bernoulli indicators
with `P(I_c = 1) = p_c`, where `p_c` is contrast `c`'s in-mask coverage fraction — the same
`prop_active` the current code already computes. Discretise the weights onto a uniform
grid and the exact convolution generalises directly:

```python
step = w.sum() / (n_bins - 1)
m = np.rint(w / step).astype(int)             # each weight as a whole number of bins
hist = np.zeros(m.sum() + 1); hist[0] = 1.0
for m_c, p_c in zip(m, prop_active):
    nxt = hist * (1 - p_c)
    nxt[m_c:] += hist[: len(hist) - m_c] * p_c
    hist = nxt
```

`histogram_bins` becomes `np.arange(len(hist)) * step`. It stays uniformly spaced, which
is what `_compute_null_montecarlo` and `nullhist_to_p` require. Cost is `O(k * n_bins)`:
27 x 10^5 is milliseconds.

Accuracy, measured against brute-force enumeration over all `2^k` outcomes (k=14,
sqrt(N) weights, tail probabilities at three thresholds):

| n_bins | grid step | max abs tail error |
|---|---|---|
| 1,001 | 1e-3 | 3.0e-3 |
| 10,001 | 1e-4 | 1.0e-5 |
| 100,001 | 1e-5 | 1.9e-15 |

Default `n_bins = 100_000`, with a guard that raises the count if `min(w) / step < 1`
(no weight may round to zero). Exposed as `StudyWeights(..., n_bins=)`? No — it is a
property of the null, not of the weights; expose it as `MKDADensity(histogram_bins=)` only
if a user ever needs it. Start with the constant and the guard.

**When the weights are uniform, the existing integer-bin path runs verbatim.** The branch
is on `np.allclose(w, w[0])`, not on whether `weighting` was passed, so `weighting=None`
and `weighting=StudyWeights()` on a corpus with equal N both produce byte-identical output
to today.

The montecarlo null needs no change beyond the bins: `_compute_permutation_summarystat`
already forwards `weight_vec_` to `KDAPlan.summary_stat`, which already accepts arbitrary
per-study weights (`permutation.py:297`). This matches CANlab: weights stay attached to
their contrast while the foci move, so the max-statistic FWE null remains valid.

Cost note: 5,000 montecarlo iterations each histogramming ~230k voxels into 10^5 bins
instead of 28 is roughly 20-30 s of added wall time. `null_method="approximate"` is the
default and is unaffected.

### 3.5 Missing data

Following CANlab exactly: after applying `transform` and `delta`, any weight that is NaN,
zero or negative is replaced by the mean of the remaining valid weights, and a warning
names the affected ids and the count. If every weight is invalid, raise. `on_missing="raise"`
turns the first case into an error for analysts who would rather not have a silent
imputation.

Imputing rather than dropping keeps the weighted and unweighted analyses over the same
study set, which is the comparison a reader will make.

### 3.6 Data flow

```
MKDADensity.__init__(weighting=...)
  -> stores self.weighting (str shorthand resolved to StudyWeights here, so
     get_params round-trips an object)

MKDADensity._preprocess_input(dataset)
  -> super()._preprocess_input(dataset)
  -> if self.weighting is not None:
         self._raw_weights_ = self.weighting.raw_weights(dataset, self.inputs_["id"])
     else:
         self._raw_weights_ = None

MKDADensity._compute_weights(ma_values)          # called from CBMAEstimator._fit
  -> ids = np.unique(self.inputs_["coordinates"]["id"].values)   # the kernel's row order
  -> raw = ones(len(ids)) if self._raw_weights_ is None
           else self._raw_weights_.reindex(ids).to_numpy()       # explicit, by id (D9)
  -> assert len(ids) == ma_values.shape[0]
  -> return (len(ids) * raw / raw.sum())[:, None]

MKDADensity._determine_histogram_bins(ma_maps)
  -> uniform weights  -> np.arange(k + 1)                        (unchanged)
  -> otherwise        -> np.arange(n_grid) * step                (§3.4)

MKDADensity._compute_null_approximate(ma_maps)
  -> uniform weights  -> today's np.convolve loop                (unchanged)
  -> otherwise        -> the shifted-convolution loop            (§3.4)

MKDADensity._prepare_subsample_null(ma_maps, subset_study_ids)
  -> unchanged: filters inputs_["coordinates"], recalls _compute_weights,
     which reindexes and renormalises over the subset (CANlab Meta_Select_Contrasts)

MKDADensity._generate_description()
  -> adds the weighting sentence and the effective sample size (§4.4)
```

Files touched:

| File | Change |
|---|---|
| `nimare/meta/cbma/weights.py` | new: `StudyWeights` |
| `nimare/meta/cbma/mkda.py` | `MKDADensity.__init__`, `_preprocess_input`, `_compute_weights`, `_determine_histogram_bins`, `_compute_null_approximate`, `_generate_description`, docstring |
| `nimare/meta/cbma/base.py` | `_p_to_summarystat` montecarlo branch (D5, separate commit) |
| `nimare/resources/references.bib` | add `wager2009evaluating` |
| `nimare/tests/test_meta_mkda.py` | §7 |
| `nimare/tests/test_meta_weights.py` | new, unit tests for `StudyWeights` |
| `docs/` | user-facing note on when weighting helps and what it assumes |

New bibliography entry:

```bibtex
@article{wager2009evaluating,
  title={Evaluating the consistency and specificity of neuroimaging data using meta-analysis},
  author={Wager, Tor D and Lindquist, Martin A and Nichols, Thomas E and Kober, Hedy and
          Van Snellenberg, Jared X},
  journal={NeuroImage},
  volume={45},
  number={1 Suppl},
  pages={S210--S221},
  year={2009},
  doi={10.1016/j.neuroimage.2008.10.061}
}
```

`MKDADensity`'s docstring currently cites `wager2007meta`, which is the SCAN review
("Meta-analysis of functional neuroimaging data: current and future directions"). The
weighting equation is in the 2009 NeuroImage paper; both should be cited.

## 4. Scientific review

### 4.1 Is `sqrt(N)` the right weight?

It is what the paper specifies and what the reference implementation computes, so
fidelity is not in question. Whether it is *optimal* is a separate matter, and the honest
answer is no one has shown it is.

Under a random-effects model with between-study variance `tau^2`, the inverse-variance
weight for a standardised effect is `1 / (sigma^2/N + tau^2)`, which is `~N` when `tau^2`
is small and flattens toward uniform as `tau^2` grows. `sqrt(N)` sits between those two
regimes. But the quantity being combined here is not an effect size — it is a binary
indicator of "did this contrast report a peak within `r` mm", which is a thresholded,
heavily censored function of the underlying statistic. No inverse-variance argument
applies to it cleanly. `sqrt(N)` is best read as "weight by the scale on which a study's
Z-statistic grows", which is the same intuition behind Stouffer's method and behind
SDM's sqrt(N) weighting.

Design consequence: expose `transform` (`"sqrt" | "linear" | "none"`) so the choice is
visible and comparable, default to `"sqrt"`, and document that `"linear"` is not the
published method.

### 4.2 The fixed-effects discount

0.75 is a convention. The paper offers no derivation, and CANlab's shipped code does not
apply it automatically despite its documentation saying it will. Bossier et al. (2018)
supports a discount's direction but not its size (§1.3).

Two failure modes the design has to avoid, and does:

- *Inferring* the label from anything (journal era, presence of a `subjects` field,
  software name). A systematic misclassification would bias every voxel. Hence D7:
  no field, no discount.
- Applying the discount while most of the corpus is unlabelled. If `inference_field`
  resolves for only part of the corpus, the unlabelled studies get `delta = 1` — i.e. they
  are treated as random-effects. That is the charitable default, but it means partial
  labelling systematically *promotes* unlabelled studies. The implementation should warn
  with the coverage fraction whenever `inference_field` resolves for less than all ids.

### 4.3 Validity of the inference under weighting

- *Approximate (uncorrected) null.* Exact given independence across contrasts and each
  contrast's own coverage fraction `p_c`; weighting changes the statistic, not those
  assumptions. The only new error is grid discretisation, bounded and measured in §3.4.
  Unchanged from today: independence across contrasts is assumed, and contrasts from the
  same study are not independent — a pre-existing MKDA limitation, not one this adds.
- *Monte Carlo FWE.* Valid. Under the null the weights are fixed constants and only the
  foci move, so the max-statistic distribution is the correct reference. Exactly CANlab's
  construction.
- *Exchangeability.* Weighting does not disturb it. The permuted map for contrast `c`
  carries contrast `c`'s weight and contrast `c`'s number of foci.

One real caveat the design should surface rather than hide: heterogeneous weights lower
the effective number of contrasts. For weights normalised to sum `k`,
`ESS = k^2 / sum(w^2)`. On the measured corpus, `ESS = 24.8` of 27 for N spanning 8-200 —
a small loss, because `sqrt` compresses hard. A corpus with one N=1000 study among twenty
N=12 studies would lose much more, and a user should be told. §4.4.

### 4.4 What the description should say

NiMARE auto-generates a methods paragraph. When weighting is on it must state the weight
(`sqrt(N)`), whether a fixed-effects discount was applied and to how many contrasts, how
many weights were imputed, and the effective sample size. A weighted meta-analysis whose
methods section does not say what was weighted is not reproducible, and the ESS is the
one number that tells a reader whether one study is driving the map.

### 4.5 What this does not fix

MKDA's statistic remains a weighted count of *thresholded* indicators. A large study that
reports one peak contributes exactly as much at that voxel as a large study that reports a
massive cluster there, and a study that fell just short of its own threshold contributes
nothing. Sample-size weighting improves how contrasts are *combined*; it does not address
what is lost in reducing each contrast to peaks. Estimators that model reported effect
magnitudes (SDM, CBMR, image-based IBMA) address a different part of the problem, and
weighting MKDA should not be described as closing that gap.

## 5. Architecture review

### 5.1 What the design gets right

- **The default path is untouched.** `weighting=None` is the default, and the null branches
  on weight uniformity rather than on whether the argument was passed. Both the "user
  didn't ask" and "user asked but N is constant" cases produce today's bytes. This is the
  property to pin with a test (§7).
- **One new module, one new concept.** `StudyWeights` names something the domain already
  has a name for. It does not introduce a parallel weighting framework for estimators that
  do not weight.
- **The seam is at a natural joint.** "Relative weights from metadata" and "normalise over
  the active subset" are genuinely different responsibilities with different lifetimes —
  the first is per-fit, the second is per-subsample. Splitting them is what makes
  leave-one-out correct without a second code path.
- **It deletes a trap.** `_compute_weights` currently reads columns that never exist. After
  this change it reads a Series that is either present or `None`.
- **It uses the existing metadata plumbing** (`_add_metadata_to_dataframe`,
  `coerced_sample_sizes`) rather than a third way to find sample sizes.

### 5.2 Risks and what mitigates them

| Risk | Mitigation |
|---|---|
| D5 (`_p_to_summarystat`) changes unweighted montecarlo cluster thresholds for MKDA *and* ALE. | Separate commit, separate justification, with before/after cluster counts on the test corpus. It is a prerequisite, not a bundle. |
| Users conflate this with ALE's sample-size handling, where `sample_size` widens the *kernel* rather than weighting the study. | Docstring says so explicitly; `reduce="sum"` vs ALE's mean is called out in §3.3. |
| `reduce="sum"` is wrong for corpora whose `sample_sizes` list is not a per-group decomposition of one contrast. | Exposed and documented; the warning on imputation surfaces the pathological cases (zero, NaN). Consider logging the N range at fit time. |
| `n_bins = 100_000` is a constant chosen from one accuracy experiment. | The experiment is recorded (§3.4) and the guard prevents the only qualitative failure (a weight rounding to zero). Revisit only if a real corpus shows error above tolerance. |
| The weighted montecarlo null is ~25 s slower. | Default null method is unaffected; document the cost. |
| `StudyWeights` invites growth into a general weighting framework (quality scores, precision weights, custom callables). | Ship `source`, `transform`, `reduce`, the FFX pair and `on_missing`. Nothing else until something asks for it. A mapping/array `source` already covers arbitrary analyst weights, which is what Wager 2009's "other study quality measures" sentence asks for. |

### 5.3 Alternatives considered and rejected

- **Flat constructor arguments** (`weight_by=`, `weight_transform=`, ...). Five parameters
  on `MKDADensity.__init__` that are inert whenever the first is `None`. Rejected on
  cohesion.
- **Weight automatically when `sample_sizes` is present.** Zero API, and it silently
  changes results for every current MKDA user on their next upgrade. Rejected.
- **Declare `sample_sizes` in `_required_inputs`.** Reuses existing machinery, but
  `PerAnalysis.validity` *drops* analyses missing the field, so switching weighting on
  would quietly shrink the corpus. Rejected in favour of D8's imputation.
- **Saddlepoint or normal approximation to the weighted Poisson binomial.** `O(k)` per
  tail evaluation and accurate far into the tail, but it replaces an exact convolution
  with an approximation of a different character and would need its own validation. The
  grid convolution strictly generalises what is already there and converges provably.
  Keep saddlepoint in reserve if grid cost ever binds.
- **Weight the MA maps themselves inside the kernel** (`ivectors .* wts` as CANlab does).
  It would require the kernel to know about study metadata and would break MA-map caching
  and reuse across estimators, since the cached maps would no longer be weight-agnostic.
  NiMARE's split — unweighted MA maps, weights applied in the summary statistic — is the
  better factoring and already exists. Keep it.

### 5.4 Resolved since the first draft

- **`reduce` defaults to `"mean"`, not `"sum"`** (D6). The first draft left this open,
  reasoning that `sample_sizes` might decompose a contrast by group. The converters
  settle it: §3.3.
- **The Monte Carlo binning cost was real, and is fixed.** `np.histogram` against an
  explicit 100k-edge array costs 12.9 ms per permutation against 2.3 ms at 28 bins.
  `MKDADensity._compute_null_montecarlo_permutation` bins the weighted grid with
  `np.bincount` on rounded indices instead, which is flat in bin count (1.2 ms) and is
  also the convention `nullhist_to_p` already uses for the observed map. That takes a
  weighted Monte Carlo fit from 2.69x the unweighted one down to 1.48x. The default
  approximate null is barely affected. §7.2.
- **An extreme weight ratio could have allocated 40 GB.** The first implementation shrank
  the grid step to fit the smallest weight, so two explicit weights of 1e-9 and 5 asked
  for five billion bins. A weight below one step now gets one bin instead, which
  overstates it by less than `step` and bounds the grid at `n_bins + k`.

## 6. Out of scope

- **`MKDAChi2`.** Wager 2009's chi-square runs on unweighted counts (§1.4), and a
  chi-square statistic computed on weighted counts does not retain its reference
  distribution — the variance of a weighted sum of Bernoullis is not the count variance
  the chi-square assumes. A weighted two-sample MKDA is a real thing, but it is CANlab's
  weighted proportion *difference* with label permutation (`meta_apply_contrast`, §1.2),
  which is a different estimator with a different null, not a weighted `MKDAChi2`.
  `MKDAChi2` already has `fwe_null_method="label-permutation"`, so the machinery is
  partly there; it is a follow-up.
- **`KDA`.** Its statistic sums kernel values rather than indicators, so the weights would
  be well-defined, but there is no published precedent for weighting it and no user asked.
- **Quality weights beyond the FFX discount.** Covered by passing an explicit mapping as
  `source`; no dedicated API.

## 7. Measurements from the implementation

### 7.1 The weighted null is correct

Against brute-force enumeration over all `2^k` outcomes, six random corpora with
`k` in 6..16 and `N` in 6..300:

| check | worst over six corpora |
|---|---|
| upper-tail probability at the 50th, 90th, 99th and 99.9th percentiles | 1.8e-10 |
| mean of the distribution | 1.1e-5 (on a statistic whose scale is `k`) |
| total mass | 1.0 to within 1e-12 |

Against 2,000,000 Monte Carlo draws of `sum_c w_c Bernoulli(p_c)`, for `k` of 50, 200 and
800, all 15 tail-probability comparisons from the median down to p=1e-4 fall within 2.2
Monte Carlo standard errors -- i.e. the analytic null is indistinguishable from sampling
the thing it claims to describe.

Equal weights reproduce today's `np.convolve` loop exactly, and a 5000:1 weight ratio on a
100-bin grid still keeps the smallest weight off zero and the mass at 1.0.

### 7.2 Performance

All figures are the minimum of repeated runs on an otherwise idle machine, on a
228,483-voxel mask with k=27 unless stated. Null construction, one-off:

| k | unweighted convolve | weighted, 100k bins |
|---|---|---|
| 27 | 0.08 ms | 1.81 ms |
| 400 | 1.13 ms | 34.0 ms |
| 3000 | 11.6 ms | 249 ms |

22-30x the unweighted convolution and still under 0.25 s at 3,000 contrasts. End to end
that is a whole fit of 28.2 ms unweighted against 35.7 ms weighted -- which is why the
grid resolution is nearly free in the default configuration, and only shows up under
`null_method="montecarlo"`.

Monte Carlo, 228,483-voxel mask, per permutation:

| configuration | ms/iteration | 500-iteration fit | vs unweighted |
|---|---|---|---|
| unweighted, 28 bins | 2.03 | 1.01 s | 1.00x |
| sqrt(N), 100k bins, `np.bincount` (ships) | 3.00 | 1.50 s | 1.48x |
| sqrt(N), 10k bins, `np.bincount` | 2.85 | 1.43 s | 1.40x |
| sqrt(N), 100k bins, `np.histogram` | 5.69 | 2.85 s | 2.69x |

Binning is no longer the bottleneck: per permutation `np.histogram` against a 100k-edge
array costs 12.9 ms where `np.bincount` costs 1.2 ms. The remaining ~1 ms/iteration gap to
the unweighted run is the cost of carrying a 100,000-element count array through the
permutation loop -- allocation, accumulation and `_get_last_bin` -- rather than a
28-element one.

Dropping to 10,000 bins recovers only 5% of that while giving up five orders of magnitude
of tail accuracy (§7.1), so 100,000 stays the default and `n_histogram_bins` is exposed for
the rare Monte Carlo run where 1.48x matters.

### 7.3 The unweighted path did not move

`stat`, `p`, `z`, `histogram_bins` and the null histogram are bit-identical to the parent
commit for `MKDADensity` and `KDA` on the bundled NIMADS studyset, compared across a git
worktree. A weighted fit of an equal-`N` corpus is bit-identical to an unweighted one, by
the short-circuit in `normalize_weights`.

## 8. Validation plan

Fidelity first, then behaviour.

1. **Unit, no imaging.** `StudyWeights.raw_weights` against hand-computed
   `delta * sqrt(N)` for: scalar `sample_size`, list `sample_sizes` at analysis level,
   study-level fallback, missing/NaN/zero (imputation value equals the mean of the valid
   weights, matching `Meta_Setup.m`), `on_missing="raise"`, FFX labels matched
   case-insensitively, partial-label warning.
2. **Equivalence gate (the one that must not fail).** `MKDADensity()` and
   `MKDADensity(weighting="sample_size")` on a corpus with constant N produce identical
   `stat`, `p`, `z` and identical `histogram_bins`. And `MKDADensity()` before and after
   the change produces identical maps against a pinned baseline.
3. **Null correctness.** For k <= 14, the weighted grid convolution matches brute-force
   enumeration over `2^k` outcomes to the tolerance in §3.4. For larger k, it matches a
   direct Monte Carlo draw of `sum w_c Bernoulli(p_c)`.
4. **Cross-implementation.** Run CANlab MKDA on a small public corpus and check that
   `k * Activation_proportion.img` equals NiMARE's weighted `stat` map. The weight vector
   is the part worth checking exactly; the kernel and mask conventions will differ and
   should be matched or documented as the source of any residual.
5. **Renormalisation under subsetting.** Leave-one-out weights equal
   `(k-1) * raw[-i] / sum(raw[-i])`, i.e. `Meta_Select_Contrasts`'s behaviour, and
   `_prepare_subsample_null` reproduces it for the diagnostics path.
6. **Alignment.** Construct a studyset whose id ordering differs from sorted order, and
   assert the weight attached to each MA-map row is that row's study's weight. This is the
   test D9 exists for; it should fail if someone reverts to positional alignment.
7. **D5 regression.** `_p_to_summarystat` returns a *monotone decreasing* threshold in `p`
   for both null methods, and matches the reverse-cumulative answer in §2.3.
