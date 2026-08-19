# GCLDA degenerates on dense (high word-token) corpora

**Date:** 2026-08-19
**Status:** Root cause established and empirically confirmed. No fix implemented.
**Scope:** Applies equally to `nimare/annotate/gclda.py` and the Rust port in
`rust/gclda/`. This is a property of the GC-LDA model as specified, **not** a
coding defect in either implementation.

---

## 1. Summary

Training GCLDA on NeuroQuery **full-text (`body`) word counts** produces a
completely degenerate model: all topics share essentially the same spatial
distribution, and all topics converge to the corpus-wide unigram distribution
(top tokens are `patients`, `activation`, `task`, `activity`, `left`, ...).

The cause is that GC-LDA's peak-token conditional contains a
word-likelihood feedback factor whose exponent scales **linearly with the number
of word tokens in the document**. GC-LDA relies on this factor being a *sparse
and informative* constraint — it restricts each peak to the small set of topics
the document's words support. When documents are long enough that their word
tokens occupy a large fraction of all topics, that constraint stops selecting
and starts randomising, and the spatial model loses all influence over topic
assignment.

**Critically, more iterations cannot help.** The spatial log-likelihood of the
failing run improved by 0.36% over 5000 iterations and was flat after roughly
1000. It is a converged degenerate mode, not slow mixing.

**And topic count is not the variable.** A matched control at T=50 (section 5)
holding peaks, seed, hyperparameters and iteration budget fixed, and varying
*only* the word-count source, reproduces the degeneracy on body text
(mean pairwise spatial correlation 0.965) while the abstract corpus reaches
0.025.

---

## 2. Reproduction

Failing run (the one that motivated this investigation):

```bash
python benchmarks/run_gclda_sweep.py \
    --nq-source body \
    --neurosynth-data-dir /mnt/c/Users/tsalo/.nimare
```

which for the 200-topic model issues:

```bash
rust/gclda/target/release/gclda-train \
    --counts gclda_runs/corpus/counts.tsv \
    --coordinates gclda_runs/corpus/coordinates.tsv \
    --mask nimare/resources/templates/MNI152_2x2x2_brainmask.nii.gz \
    --out-dir gclda_runs/topics-200 \
    --n-topics 200 --n-regions 2 --symmetric true \
    --n-iters 5000 --loglikely-freq 50 --seed-init 1 --threads 8
```

Model hyperparameters were the NiMARE defaults throughout:
`alpha=0.1, beta=0.01, gamma=0.01, delta=1.0, dobs=25, roi_size=50.0`,
`n_regions=2`, `symmetric=true`, `seed_init=1`.

### Corpora

Both corpora are NeuroQuery v1, `neuroquery7547` vocabulary, **real integer
`type-count` features** (not tf-idf). They share identical documents, identical
vocabulary, and an identical peak set — they differ *only* in how many word
tokens each document carries.

| | `--nq-source abstract` | `--nq-source body` | ratio |
|---|---:|---:|---:|
| documents | 13,459 | 13,459 | 1.00x |
| vocabulary terms | 7,547 | 7,547 | 1.00x |
| peak tokens | 469,260 | 469,260 | 1.00x |
| **word tokens** | **1,157,059** | **21,661,055** | **18.72x** |
| non-zero count cells | 712,311 | 5,620,594 | 7.89x |
| word tokens / doc (median) | 85 | 1,532 | 18.0x |
| word tokens / doc (mean) | 86.0 | 1,609.4 | 18.7x |
| peak tokens / doc (median) | 22 | 22 | 1.00x |
| **words : peaks** | **2.47 : 1** | **46.16 : 1** | **18.7x** |

---

## 3. The mechanism

### 3.1 The conditional

In the peak-token sampler
(`nimare/annotate/gclda.py:293-296`, mirrored bit-for-bit at
`rust/gclda/src/sampler/peaks.rs:212-215`), the unnormalised weight for
assigning peak token *i* in document *d* to (topic *t*, region *r*) is:

```
probs_pdf[r, t] =  peak_probs[i, t, r]                                  # spatial PDF  N(x_i | mu_tr, Sigma_tr)
                 * (region_by_topic[r, t] + delta)                      # region prior
                   / (region_totals[t] + delta * n_regions)
                 * (doc_by_topic[d, t] + alpha)                         # LDA doc-topic term for the peak itself
                 * peak_topic_probs[t]                                  # <-- the word-likelihood feedback factor
```

where

```
peak_topic_probs[t] = exp( logp[t] - max_t' logp[t'] )
logp[t]             = n_word_doc[d, t] * log1p( 1 / (n_peak_doc[d, t] + gamma) )
```

`n_peak_doc` has already been decremented for the peak token being resampled.

### 3.2 The factor is mathematically correct

GC-LDA's correspondence assumption is that a document's *word* topics are drawn
from its *peak*-derived document-topic distribution:

```
p(z = t | d) = (n_peak_doc[d, t] + gamma) / (N_peaks_d + gamma * T)
```

Moving one peak into topic *t* raises `n_peak_doc[d, t]` by one, which changes
the joint likelihood of every word token in that document. The denominator
`N_peaks_d + gamma*T` is invariant to *which* topic the peak lands in (the total
peak count of the document is fixed), so it cancels, and the relative weight is
exactly

```
  prod over word tokens in d assigned to t of  (n_peak_doc[d,t] + 1 + gamma) / (n_peak_doc[d,t] + gamma)
= ( (n[t] + 1 + gamma) / (n[t] + gamma) ) ^ n_word_doc[d, t]
```

whose log is `n_word_doc[d,t] * log1p(1 / (n[t] + gamma))`. This is precisely
the code. **The implementation is faithful to the model.**

Provenance: the term predates the Rust port and predates the current numba
kernels. `git log -S log1p -- nimare/annotate/gclda.py` attributes it to
`cdea6dd` ("[ENH] optimize gclda and lda (#982)"), and the code that commit
*replaced* had the identical structure
(`logp = n_word_tokens_doc_by_topic[doc,:] * np.log(...)`, then
`exp(logp - max(logp))`). The Rust port reproduces Python bit-for-bit
(`nimare/tests/test_gclda_rust.py::test_rust_matches_python_every_iteration`).

### 3.3 What actually goes wrong

The factor is **effectively one-hot in both the healthy and the failing
regime.** This was initially mistaken for the pathology; it is not.

Measured median within-document gap between the best and second-best topic's
exponent, at the end of training:

| | max - min | max - 2nd | runner-up weight | docs where runner-up underflows f64 (gap > 745) |
|---|---:|---:|---:|---:|
| abstract T=50 | 124.6 nats | **41.5 nats** | 9.14e-19 | 0.0% |
| body T=50 | 1559.9 nats | **660.0 nats** | 0.00e+00 (literal underflow) | **47.0%** |
| body T=200 | 1693.7 nats | **793.8 nats** | 0.00e+00 (literal underflow) | **51.7%** |

Effective number of topics surviving the factor (`sum_t exp(logp[t] - max)`) is
median **1.000** in *all three* runs, and the winner is a topic with **zero**
peaks in that document 99.6% (abstract T=50), 92.4% (body T=50) and 99.0%
(body T=200) of the time — because `log1p(1/gamma) = log(101) = 4.615` dwarfs
`log1p(1/1.01) = 0.688`.

So in every regime the factor performs a hard, deterministic selection, and the
spatial PDF, the region prior and the alpha term only choose the *sub-region*
within the one topic the word term dictates. A fix aimed purely at softening
this factor (making it non-one-hot) would therefore be aimed at the wrong
target -- the healthy run is one-hot too.

**The discriminator is the size of the document's word support set.** A topic
with no word tokens in the document has exponent `0 * 4.615 = 0`, the minimum,
so the argmax is always restricted to topics that the document's words actually
occupy. That restriction is the correspondence constraint. Its usefulness
depends entirely on how many topics it leaves in play:

| | words/doc | topics touched by that doc's words | fraction of T | expected at random init |
|---|---:|---:|---:|---:|
| abstract T=50 | 85 | **9 of 50** | **18.0%** | 41 of 50 (82.0%) |
| body T=50 | 1,532 | **35 of 50** | **70.0%** | 50 of 50 (100.0%) |
| body T=200 | 1,532 | **98 of 200** | **49.0%** | 200 of 200 (100.0%) |

The `body T=50` row is the matched-topic-count control of section 5 and is the
important one: at the *same* topic count, the abstract corpus drives the support
set to 18% while the body corpus stalls at 70%. The failure therefore tracks
document length, not topic count.

With abstracts the sampler drives the support set from 82% down to 18% — the
constraint selects one topic from ~9 genuine candidates, different documents
support different small sets, and the spatial Gaussians differentiate.

With body text the support set only falls from 100% to 49%. 1,532 tokens spread
over 200 topics put ~7.7 word tokens in *every* topic at initialisation, so
there is no sparsity for the sampler to amplify; the argmax among ~98 near-tied
candidates is decided by Poisson noise in the word counts. The constraint
scatters peaks rather than steering them.

### 3.4 The resulting degenerate fixed point

Because the winner is always a *zero-peak* topic, the factor is additionally an
**anti-concentration ratchet**: it deals a document's peaks out roughly
one-per-topic across its highest-word-count topics. Observed in the failing
run: **median 15 distinct topics for a document's 22 peaks.**

That closes a self-sustaining loop:

1. Peaks are scattered near-uniformly across topics within each document.
2. `_update_regions` therefore fits each (topic, region) Gaussian to a
   near-random subset of the whole peak cloud, yielding the grand mean and grand
   covariance for every topic.
3. Those identical, near-whole-brain Gaussians make the spatial PDF completely
   uninformative — but it was already irrelevant per 3.3.
4. The word sampler's document prior is `n_peak_doc[d,t] + gamma`, which is now
   flat, so word assignment falls back on the global `p(w|t)` term alone and
   every topic drifts toward the corpus unigram distribution.

---

## 4. Measured symptoms (body, T=200, 5000 iterations)

Model at `gclda_runs/topics-200/`.

**Spatial collapse**

- Every region Gaussian sits at the grand mean of the peak cloud:
  mu ~ (+/-29.0, -18.6, +14) mm; the across-topic standard deviation of mu is
  only **(5.1, 4.0) mm in y and z** (the x figure of 28.7 mm is an artifact of
  the `symmetric=true` mirroring, not real variation).
- Mean region standard deviations **(19.5, 39.3, 26.0) mm** — near-whole-brain.
- Mean pairwise correlation of `p(voxel|topic)` columns: **0.899**
  (median 0.931, max 0.998).

**Uniform topic occupancy** (the signature of near-random assignment)

- Peak tokens per topic: min 1,310 / median 2,210 / max 5,234, against
  469,260 / 200 = 2,346 for perfectly uniform.
- Word tokens per topic: min 87,043 / median 106,946 / max 143,190, against
  21,661,055 / 200 = 108,305 for perfectly uniform.

**Word non-specialisation**

- Topic word entropy: median **5.85 nats** against **8.90** for uniform over the
  7,300 realised vocabulary terms.
- Only **57 distinct top-1 words across 200 topics**:
  `patients` (20 topics), `activation` (18), `task` (16), `activity` (14),
  `left` (13), `brain` (11), `group` (8), `right` (6).

**Log-likelihood: converged, not under-trained**

| | iter 0 | final | gain |
|---|---:|---:|---:|
| spatial `x` | -6,905,927.24 | -6,881,417.56 (iter 5000) | **+0.36%** |
| word `w` | -151,350,122.29 | -145,786,293.29 (iter 5000) | +3.68% |

The spatial term is flat from roughly iteration 1000. The word term was still
falling near-linearly at iteration 5000 — 21.7M tokens over 200 topics had not
converged either, which is a separate (and much more expensive) problem.

---

## 5. Controlled experiment

Identical binary, identical peaks, identical hyperparameters, identical seed;
**only the word-count source differs.**

```bash
B=rust/gclda/target/release/gclda-train
M=nimare/resources/templates/MNI152_2x2x2_brainmask.nii.gz

# abstract
$B --counts gclda_ctrl/corpus_abstract/counts.tsv \
   --coordinates gclda_ctrl/corpus_abstract/coordinates.tsv --mask $M \
   --out-dir gclda_ctrl/abs-T50  --n-topics 50 --n-regions 2 --symmetric true \
   --n-iters 300 --loglikely-freq 25 --seed-init 1 --threads 4

# body
$B --counts gclda_runs/corpus/counts.tsv \
   --coordinates gclda_runs/corpus/coordinates.tsv --mask $M \
   --out-dir gclda_ctrl/body-T50 --n-topics 50 --n-regions 2 --symmetric true \
   --n-iters 300 --loglikely-freq 25 --seed-init 1 --threads 4
```

Note that the spatial log-likelihood at iteration 0 is essentially identical for
both (-6,905,368.80 abstract vs -6,905,927.24 body) — same peaks, same random
init — so the trajectories are directly comparable.

The first two columns are the matched pair and differ in **exactly one input**:

| | abstract T=50, 300 it | body T=50, 300 it | body T=200, 5000 it |
|---|---:|---:|---:|
| word tokens / doc (median) | 85 | 1,532 | 1,532 |
| peak tokens / doc (median) | 22 | 22 | 22 |
| **spatial log-lik gain** | **+4.69%** | **+0.26%** | **+0.36%** |
| word log-lik gain | +10.17% | +2.44% | +3.68% |
| **mean pairwise spatial corr** | **0.025** | **0.965** | **0.899** |
| **mu spread across topics (y, z)** | **27.7, 17.2 mm** | **3.7, 2.5 mm** | **5.1, 4.0 mm** |
| mean region SD | 16.7, 25.8, 18.6 mm | 19.7, 39.7, 26.4 mm | 19.5, 39.3, 26.0 mm |
| topics touched by a doc's words | 9 of 50 (18%) | 35 of 50 (70%) | 98 of 200 (49%) |
| peak topics per doc (of 22 peaks) | 7 | 16 | 15 |
| feedback gap, max - 2nd | 41.5 nats | 660.0 nats | 793.8 nats |
| feedback-factor effective topics | 1.000 | 1.000 | 1.000 |
| winner has zero peaks in doc | 99.6% | 92.4% | 99.0% |

The abstract model reaches near-orthogonal, spatially distinct topics
(corr 0.025) in **300** iterations. The body model at the **same topic count,
same peaks, same seed, same iteration budget** is fully degenerate
(corr 0.965), and the T=200 / 5000-iteration run is no better.

Two conclusions follow that the T=200-only evidence could not establish:

1. **Topic count is not the variable.** Body text is degenerate at T=50 and at
   T=200 alike; abstracts are healthy at T=50. Document length is the cause.
2. **The degeneracy is present from the start, not drifted into.** Body T=50 is
   already at corr 0.965 after 300 iterations, so the chain never leaves the
   initial scattered configuration rather than slowly decaying into it. This
   matters for candidate fix 8.5 (annealing): there is no good early state to
   preserve.

---

## 6. Ruled out

- **Not a Rust port bug.** The Rust sampler reproduces the Python reference
  bit-for-bit under `test_rust_matches_python_every_iteration`, and the offending
  expression is textually identical in both.
- **Not a regression from `cdea6dd`.** The replaced code had the same structure.
- **Not too few iterations.** See section 4; the spatial term is flat.
- **Not a plotting artifact.** `benchmarks/report_gclda.py` is faithful; the
  degeneracy is present in the raw `regions_mu.npy` / `regions_sigma.npy` /
  `p_voxel_g_topic.npy` arrays.
- **Not the tf-idf proxy problem.** Both corpora here carry genuine NeuroQuery
  integer `type-count` features.

---

## 7. Diagnostic metrics for evaluating any future fix

These are the statistics that separated the two regimes, in rough order of
diagnostic value. Any candidate fix should move the body-corpus numbers toward
the abstract column of section 5.

1. **Word support-set size** — median over documents of
   `(n_word_tokens_doc_by_topic > 0).sum(axis=1) / n_topics`.
   Healthy ~18%; degenerate ~49%. This is the most direct predictor.
2. **Mean pairwise correlation of `p(voxel|topic)` columns** (mean-centred,
   L2-normalised). Healthy ~0.03; degenerate ~0.90.
3. **Across-topic spread of `regions_mu`** in y and z (skip x under
   `symmetric=true`). Healthy ~28 / 17 mm; degenerate ~5 / 4 mm.
4. **Spatial log-likelihood gain** from iteration 0. Healthy: several percent
   within a few hundred iterations. Degenerate: <0.5% over thousands.
5. **Feedback-factor effective topic count** —
   `sum_t exp(logp[t] - max_t logp[t])`. Note this is ~1.0 even when healthy,
   so it is only useful alongside metric 1; and the **max-minus-second gap** in
   nats is the more sensitive form (41 nats healthy vs 794 nats degenerate).
6. **Within-document peak spread** — distinct topics used by a document's peaks
   versus its peak count. Degenerate: 15 topics for 22 peaks (round-robin).
7. **Topic occupancy uniformity** — spread of peak/word tokens per topic against
   `total / n_topics`.

Working scripts used to compute all of the above are reproduced in appendix A.

---

## 8. Candidate directions (none implemented, none validated)

Ordered by my estimate of principle-to-effort ratio. All predictions are
reasoned, **not measured** — they are hypotheses to test, and each should be
evaluated against the section 7 metrics.

### 8.1 Down-weight the feedback exponent by document length

Replace `n_word_doc[d, t]` with `lambda_d * n_word_doc[d, t]`, choosing
`lambda_d = L_ref / N_words_d` for a reference document length `L_ref`
(e.g. 85, the abstract median). This makes the factor's magnitude invariant to
document length.

- *Predicted:* directly restores the abstract-regime exponent scale. Should be
  the single most effective lever.
- *Caveat:* this is no longer the exact collapsed conditional for GC-LDA. It
  corresponds to a valid generative model in which the document's word tokens
  are replaced by a down-weighted pseudo-count — the standard modality-balancing
  trick in multimodal topic models — and should be documented as a deliberate
  model change, not presented as the same model.
- *Open question:* does rescaling the exponent alone fix it, given that the
  support set (metric 1) is a property of the *word* sampler and may not shrink
  just because the peak sampler is gentler? This is the crux and should be
  tested first. My expectation is that it partially self-corrects, because a
  less scattered peak distribution sharpens the word sampler's `n_peak_doc + gamma`
  document prior, which in turn concentrates the words — but that is exactly the
  loop in 3.4 running in reverse, and it may need help from 8.2 or 8.3.

### 8.2 Subsample word tokens per document

Cap each document at `L` word tokens (e.g. 100), sampled without replacement,
before constructing `wtoken_*` arrays. Leaves the sampler untouched.

- *Predicted:* reproduces the abstract regime by construction, and is
  dramatically cheaper (word sampling is O(word tokens x T) and dominates
  runtime on dense corpora — an estimated ~73% of per-iteration time at T=400 on
  body, extrapolated by `run_gclda_sweep.py`'s phase-scaling model rather than
  measured directly).
- *Caveat:* discards data; would want several subsample replicates / chains and
  a stability check across them.
- Cheapest thing to try, and a useful control for 8.1: if 8.2 works and 8.1 does
  not, the support set is the binding constraint rather than the exponent scale.

### 8.3 Prune the vocabulary to discriminative terms

Drop very high document-frequency terms (`patients`, `task`, `activation`, ...).
Reduces tokens per document and raises sparsity simultaneously.

- *Predicted:* helps on both axes but probably insufficient alone; the target is
  a support set well under ~20% of T, and body text would likely need very
  aggressive pruning to get there from 49%.
- Compose with 8.1 or 8.2 rather than relying on it.

### 8.4 Raise `gamma`

`gamma = 0.01` gives `log1p(1/gamma) = 4.615` for a zero-peak topic against
`0.688` for a one-peak topic — a 6.7x preference that drives the
anti-concentration ratchet of 3.4. `gamma = 1.0` gives `0.693` vs `0.405`,
a 1.7x preference.

- *Predicted:* weakens the ratchet, so peaks may concentrate rather than
  round-robin. Does **not** address the exponent magnitude
  (1,532 x ~0.5 is still enormous), so expect partial improvement at best.
- *Caveat:* `gamma` also appears in the word sampler's document prior, so this
  is not an isolated change.

### 8.5 Tempering / annealing on the feedback exponent

Ramp `lambda` from 0 (spatial-only) to 1 over training, letting the spatial
model organise before the correspondence constraint takes hold.

- *Predicted:* may help escape the degenerate basin, but if `lambda = 1` is
  itself degenerate for this corpus the chain will simply fall back into it.
  Worth trying only in combination with 8.1.

### 8.6 Relax hard correspondence to soft (model change)

Give word tokens their own document-topic distribution `theta_d^w` tied to the
peak-derived one through a Dirichlet with a concentration parameter, instead of
drawing word topics directly from `n_peak_doc[d,:] + gamma`. The concentration
parameter then explicitly controls how strongly the two modalities are coupled,
and can be fit or cross-validated rather than being implicitly set by document
length.

- *Predicted:* the principled long-term answer; makes the coupling strength a
  first-class, tunable quantity instead of an accident of corpus density.
- *Caveat:* a genuine change to the generative model, requiring a new derivation
  of the conditionals, a new sampler, and revalidation of the decoding/encoding
  paths. It would break the bit-exactness contract with the existing Python
  implementation, so it belongs behind a new model class rather than as a change
  to `GCLDAModel`.

### 8.7 Not viable

- **More iterations** — ruled out in section 4.
- **More peaks per document** — data-limited; cannot be increased.

---

## 9. Practical guidance for now

Use abstract-length text. GC-LDA was designed and validated by Rubin et al.
(2017) on Neurosynth abstracts, and a 46:1 word-to-peak ratio is far outside
that regime. On NiMARE's NeuroQuery path that means `--nq-source abstract`
(the default in `benchmarks/run_gclda_sweep.py`), which is also ~18.7x less
word-token work.

As a rule of thumb pending a proper study: keep median word tokens per document
low enough that a document's words occupy well under ~20% of the topics at
convergence. Abstracts at T=50 sit at 18% and are healthy; body text sits at
70% at T=50 and 49% at T=200 and is degenerate at both. Note this is a joint
constraint on document length **and** topic count, so it should be re-checked
whenever either changes -- and note from section 5 that raising T does **not**
rescue a corpus that is too dense, even though it lowers the percentage.

---

## Appendix A — diagnostic scripts

Both scripts reconstruct `n_word_tokens_doc_by_topic` from the saved
`wtoken_topic_idx.npy` plus the staged `counts.tsv`, since GCLDA's word tokens
are laid out as `np.repeat` over the count matrix in row-major order.

### A.1 Support set, feedback factor, spatial degeneracy

```python
import json, numpy as np, pandas as pd
from pathlib import Path

def report(model_dir, counts_tsv, gamma=0.01):
    d = Path(model_dir)
    npt = np.load(d / "n_peak_tokens_doc_by_topic.npy").astype(np.float64)
    wtok_t = np.load(d / "wtoken_topic_idx.npy")
    counts = pd.read_csv(counts_tsv, sep="\t", index_col=0).to_numpy()
    doc_of = np.repeat(np.arange(counts.shape[0]), counts.sum(1))
    assert doc_of.size == wtok_t.size
    D, T = npt.shape
    nwd = np.zeros((D, T))
    np.add.at(nwd, (doc_of, wtok_t), 1.0)

    ll = pd.read_csv(d / "loglikelihood.tsv", sep="\t")
    x0, x1 = ll["x"].iloc[0], ll["x"].iloc[-1]

    logp = nwd * np.log1p(1.0 / (npt + gamma))
    s = np.sort(logp, 1)
    ess = np.exp(logp - s[:, -1:]).sum(1)
    win = logp.argmax(1)

    sig = np.load(d / "regions_sigma.npy").reshape(-1, 3, 3)
    mu = np.load(d / "regions_mu.npy").reshape(-1, 3)
    pvt = np.load(d / "p_voxel_g_topic.npy")
    X = pvt - pvt.mean(0, keepdims=True)
    X /= np.linalg.norm(X, axis=0, keepdims=True)
    C = X.T @ X
    iu = np.triu_indices(T, 1)

    return {
        "T": T,
        "words_per_doc": np.median(nwd.sum(1)),
        "peaks_per_doc": np.median(npt.sum(1)),
        "support_set_frac": np.median((nwd > 0).sum(1)) / T,      # metric 1
        "spatial_corr": np.median(C[iu]),                          # metric 2
        "mu_yz_spread": np.round(mu[:, 1:].std(0), 1),             # metric 3
        "x_ll_gain_pct": 100 * (x1 - x0) / abs(x0),                # metric 4
        "feedback_ess": np.median(ess),                            # metric 5
        "gap_max_2nd_nats": np.median(s[:, -1] - s[:, -2]),        # metric 5 (sensitive form)
        "win_has_zero_peaks": (npt[np.arange(D), win] == 0).mean(),
        "region_sd_mm": np.round(
            np.sqrt(np.array([m.diagonal() for m in sig])).mean(0), 1
        ),
        "peak_topics_per_doc": np.median((npt > 0).sum(1)),        # metric 6
    }
```

### A.2 Topic occupancy and word specialisation

```python
nwt = np.load(d / "n_word_tokens_word_by_topic.npy")
npt = np.load(d / "n_peak_tokens_doc_by_topic.npy")
pwt = np.load(d / "p_word_g_topic.npy")
vocabulary = [w for w in (d / "vocabulary.txt").read_text().split("\n") if w]

wtok, ptok = nwt.sum(0), npt.sum(0)                   # metric 7
P = pwt / pwt.sum(0, keepdims=True)
entropy = -(P * np.log(P + 1e-300)).sum(0)            # compare against log(len(vocabulary))

from collections import Counter
top1 = Counter(vocabulary[i] for i in np.argmax(pwt, 0))
```

---

## Appendix B — artifacts

None of these are tracked by git. `gclda_runs/` was **deliberately removed on
2026-08-19**, once the measurements below had been taken, to free disk space for
the replacement abstract-corpus sweep. Every number in this document was
recorded beforehand and is reproducible from section 2. `gclda_ctrl/` was kept.

| path | status | contents |
|---|---|---|
| `gclda_runs/topics-200/` | **deleted** | the failing body-corpus model, T=200, 5000 iterations |
| `gclda_runs/corpus/` | **deleted** | staged body-corpus `counts.tsv` / `coordinates.tsv` / `corpus_stats.json` (~204 MB) |
| `gclda_ctrl/corpus_abstract/` | present | staged abstract corpus, same documents and peaks |
| `gclda_ctrl/abs-T50/` | present | abstract control, T=50, 300 iterations |
| `gclda_ctrl/body-T50/` | present | body control, T=50, 300 iterations |
| `benchmarks/run_gclda_sweep.py` | tracked | sweep runner (`--nq-source`, `--corpus`) |
| `benchmarks/report_gclda.py` | untracked | per-topic HTML report (spatial map + top tokens) |

Re-staging the body corpus costs one `run_gclda_sweep.py --nq-source body`
invocation (the NeuroQuery download itself is cached under
`--neurosynth-data-dir`). The matched T=50 control in `gclda_ctrl/` is
sufficient to reproduce the central result without it.

Two further body runs, T=100 and T=400, were stopped mid-flight (at iterations
1650 and 2950 respectively) and never wrote `model.json`. Their partial logs,
removed with the rest of `gclda_runs/`, showed the same dead-flat spatial
log-likelihood: T=400 moved
from -6,879,820.8 at iteration 2900 to -6,879,821.9 at 2950, and T=100 from
-6,884,843.8 at 1600 to -6,884,855.6 at 1650 -- i.e. drifting *backwards* within
noise at both topic counts, consistent with sections 4 and 5.
