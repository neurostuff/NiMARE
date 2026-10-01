# NiMARE agent skills

Agent skills for running **statistically defensible neuroimaging meta-analyses** with
[NiMARE](https://github.com/neurostuff/NiMARE).

NiMARE makes it easy to run *an* analysis. These skills exist because the easy path is
frequently not the correct one: the unit of analysis is the contrast rather than the
subject sample, no minimum study count is enforced, `FWECorrector` defaults to Bonferroni
rather than the Monte Carlo cluster correction the ALE literature validates, and
decoding correlations come with degrees of freedom that make every term "significant".
Each skill carries the method *and* the guardrails.

The skill set is shaped by what people actually publish with NiMARE — a survey of 98
citing articles with full text in PubMed Central — not by the shape of the API. See
[`docs/use-case-survey.md`](docs/use-case-survey.md).

Targets **NiMARE >= 0.22**.

## The skills

| Skill | Use it for |
|---|---|
| [`nimare-meta-analysis`](skills/nimare-meta-analysis/) | Entry point: pick a method, apply the guardrails, route to the rest |
| [`nimare-dataset-curation`](skills/nimare-dataset-curation/) | Getting coordinates in, and auditing contrast independence |
| [`nimare-cbma-ale`](skills/nimare-cbma-ale/) | ALE, MKDA, KDA, SCALE, CBMR over published peaks |
| [`nimare-contrast-conjunction`](skills/nimare-contrast-conjunction/) | Subtraction, balanced subtraction, MKDA chi-square, conjunction |
| [`nimare-macm`](skills/nimare-macm/) | Meta-analytic coactivation modeling from a seed ROI |
| [`nimare-functional-decoding`](skills/nimare-functional-decoding/) | "What is this map?" against Neurosynth / NeuroQuery / BrainMap |
| [`nimare-ibma`](skills/nimare-ibma/) | Image-based meta-analysis, NeuroVault, multi-site pooling |
| [`nimare-reporting`](skills/nimare-reporting/) | Methods text, PRISMA, checklists, sharing |

Two are runnable on their own:

```bash
# Is any subject sample being counted twice?
python skills/nimare-dataset-curation/scripts/audit_independence.py foci.txt

# ALE with the settings the literature supports, not the constructor defaults
python skills/nimare-cbma-ale/scripts/run_ale.py foci.txt -o results/ --n-iters 5000
```

## Install

### Claude Code (plugin)

```
/plugin marketplace add neurostuff/nimare-skills
/plugin install nimare-skills@nimare-skills
```

### Claude Code or Claude Desktop (skills only)

```bash
git clone https://github.com/neurostuff/nimare-skills
cp -r nimare-skills/skills/* ~/.claude/skills/          # user-wide
# or, per project:
cp -r nimare-skills/skills/* your-project/.claude/skills/
```

### claude.ai

Upload any `skills/<skill-name>/` directory as a skill in Settings → Capabilities.

### Other agents (Codex, Cursor, Copilot, OpenAI Agents SDK, ...)

Each skill is a plain `SKILL.md` with YAML frontmatter plus reference Markdown and
Python — no Claude-specific runtime. Point the agent at this repository and at
[`AGENTS.md`](AGENTS.md), or vendor `skills/` into your project.

## What is in a skill

```
skills/nimare-cbma-ale/
  SKILL.md            # loaded when the task matches `description`
  references/         # read on demand, not loaded up front
  scripts/            # runnable, tested
```

## Verification

Behavioural claims were checked against NiMARE 0.22 source and by running it, not from
documentation. Notably:

- CBMA ids are `"<study_id>-<analysis_id>"` and every estimator consumes them, so
  same-sample contrasts count as independent experiments. Measured on NiMARE's bundled
  `semantic_knowledge_children.txt`: entering each sample twice took supra-threshold
  voxels from 1,587 to 3,960 and voxels above z=5 from 87 to 420, with no new data.
- `audit_independence.py` run against that same bundled file finds two pairs of study
  ids that are probably one paper each (`arnoldussen2006nc`/`arnoldussen2006rm`, both
  N=11).
- Map and table keys (`z_desc-mass_level-cluster_corr-FWE_method-montecarlo`,
  `..._diag-Jackknife_tab-counts_tail-positive`) are copied from real runs.
- Decoder signatures, return columns and the fact that `Decoder.fit()` returns `None`
  were checked by executing them.

Where NiMARE's behaviour and the methods literature disagree, the skills say so and
recommend the literature.

## Contributing

Corrections welcome, especially where NiMARE's behaviour has changed. Please note the
NiMARE version you checked against.

## License

MIT, matching NiMARE.

## Citing

Cite NiMARE itself:

> Salo, T., Yarkoni, T., Nichols, T. E., Poline, J.-B., Bilgel, M., Bottenhorn, K. L.,
> et al. (2023). NiMARE: Neuroimaging Meta-Analysis Research Environment.
> *Aperture Neuro*, 3. https://doi.org/10.52294/001c.87681

and the algorithm you used (ALE, MKDA, Neurosynth, ...).
