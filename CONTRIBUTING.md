# Contributing to NiMARE

Welcome to the ``NiMARE`` repository!
We're excited you're here and want to contribute.

These guidelines are designed to make it as easy as possible to get involved.
If you have any questions that aren't discussed below, please let us know by opening an [issue][link_issues]!

Before you start you'll need to set up a free [GitHub][link_github] account and sign in.
Here are some [instructions][link_signupinstructions].

## Governance

Governance is a hugely important part of any project.
It is especially important to have clear process and communication channels for open source projects that rely on a distributed network of volunteers, such as ``NiMARE``.

``NiMARE`` is currently supported by a small group of core developers.
Even with only a couple of individuals involved in decision making processes, we've found that setting expectations and communicating a shared vision has great value.

By starting the governance structure early in our development, we hope to welcome more people into the contributing team.
We are committed to continuing to update the governance structures as necessary.
Every member of the ``NiMARE`` community is encouraged to comment on these processes and suggest improvements.

As the first project leader, Taylor Salo is ultimately responsible for any major decisions pertaining to ``NiMARE`` development.
However, all potential changes are explicitly and openly discussed in the described channels of communication, and we strive for consensus amongst all community members.

## Code of conduct

All ``NiMARE`` community members are expected to follow our [code of conduct](https://github.com/neurostuff/NiMARE/blob/main/CODE_OF_CONDUCT.md) during any interaction with the project.
That includes- but is not limited to- online conversations, in-person workshops or development sprints, and when giving talks about the software.

As stated in the code, severe or repeated violations by community members may result in exclusion from collective decision-making and rejection of future contributions to the ``NiMARE`` project.

## Asking questions about using NiMARE

Please direct usage-related questions to [NeuroStars][link_neurostars], with [the "Software Support" category and the "nimare" tag][link_neurostars_nimare].
The ``NiMARE`` developers follow NeuroStars, and will be able to answer your question there.

## Labels

The current list of labels are [here][link_labels] and include:

* [![Good First Issue](https://img.shields.io/badge/-good%20first%20issue-7057ff.svg)](https://github.com/neurostuff/NiMARE/labels/good%20first%20issue)
*These issues contain a task that a member of the team has determined should require minimal knowledge of the existing codebase, and should be good for people new to the project.*
If you are interested in contributing to NiMARE, but aren't sure where to start, we encourage you to take a look at these issues in particular.

* [![Help Wanted](https://img.shields.io/badge/-help%20wanted-33aa3f.svg)](https://github.com/neurostuff/NiMARE/labels/help%20wanted)
*These issues contain a task that a member of the team has determined we need additional help with.*
If you feel that you can contribute to one of these issues, we especially encourage you to do so!

* [![Bug](https://img.shields.io/badge/-bug-ee0701.svg)](https://github.com/neurostuff/NiMARE/labels/bug)
*These issues point to problems in the project.*
If you find new a bug, please give as much detail as possible in your issue, including steps to recreate the error.
If you experience the same bug as one already listed, please add any additional information that you have as a comment.

* [![Enhancement](https://img.shields.io/badge/-enhancement-84b6eb.svg)](https://github.com/neurostuff/NiMARE/labels/enhancement)
*These issues are asking for new features to be added to the project.*
Please try to make sure that your requested feature is distinct from any others that have already been requested or implemented.
If you find one that's similar but there are subtle differences please reference the other request in your issue.

## Making a change

We appreciate all contributions to NiMARE, but those accepted fastest will follow a workflow similar to the following:

**1. Comment on an existing issue or open a new issue referencing your addition.**

This allows other members of the NiMARE development team to confirm that you aren't overlapping with work that's currently underway and that everyone is on the same page with the goal of the work you're going to carry out.

[This blog][link_pushpullblog] is a nice explanation of why putting this work in up front is so useful to everyone involved.

**2. Fork NiMARE.**

[Fork][link_fork] the [NiMARE repository][link_nimare] to your profile.

This is now your own unique copy of NiMARE.
Changes here won't effect anyone else's work, so it's a safe space to explore edits to the code!

Make sure to [keep your fork up to date][link_updateupstreamwiki] with the main repository.

**3. Make the changes you've discussed.**

Try to keep the changes focused. We've found that working on a [new branch][link_branches] makes it easier to keep your changes targeted.

When you're creating your pull request, please do your best to follow NiMARE's preferred style conventions.
Namely, documentation should follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/) convention and code should adhere to [PEP8](https://www.python.org/dev/peps/pep-0008/) as much as possible.

**4. Submit a pull request.**

Submit a [pull request][link_pullrequest].

A member of the development team will review your changes to confirm that they can be merged into the main codebase.

Please use a sentence-case title for the pull request, and do not include any prefixes (e.g., ``[ENH]``), as we now use labels to distinguish pull request types.
The title should summarize the changes proposed in the pull request, with an emphasis on readability, as pull request titles are used directly in our release notes.

## Development conventions

Use these conventions as a reference when adding or reviewing code.

### Code organization and module hierarchy

NiMARE is organized as a domain-focused Python package under `nimare/`.

- Core infrastructure lives in package-level modules such as `nimare/base.py`,
    `nimare/estimator.py`, `nimare/results.py`, `nimare/utils.py`, and `nimare/transforms.py`.
- Analysis methods are grouped by domain. For example, `nimare/meta/` contains meta-analytic
    estimators, with coordinate-based methods in `nimare/meta/cbma/`, image-based methods in
    `nimare/meta/ibma.py`, kernels in `nimare/meta/kernel.py`, CBMR in `nimare/meta/cbmr/`, and
    CBMR torch models in `nimare/meta/cbmr/model.py`.
- Public APIs are exposed from package `__init__.py` files. Optional CBMR exports are lazily
    imported in `nimare/meta/__init__.py` so importing `nimare.meta` does not require torch.
- Tests live in `nimare/tests/` and are grouped by feature or module, for example
    `test_meta_ale.py`, `test_meta_ibma.py`, and `test_meta_cbmr.py`.
- Examples and benchmarks are kept outside the package in `examples/` and `benchmarks/`.

### Naming conventions

- Modules and files use lowercase, descriptive names with underscores when needed
    (`cbmr.py`, `models.py`, `test_meta_cbmr.py`).
- Classes use `PascalCase` (`CBMR`, `CBMRModel`, `CBMRResult`).
- Functions, methods, variables, and pytest fixtures use `snake_case`.
- Private helpers and implementation details are prefixed with an underscore
    (`_validate_incidence_threshold`, `_build_group_foci_matrices`).
- Constants use uppercase names (`DEFAULT_GROUP_NAME`, `DEFAULT_FLOAT_DTYPE`).
- Tests are named `test_<behavior>` and should describe the behavior under test rather than the
    implementation detail alone.

### Design patterns and abstractions

- Estimators inherit from `Estimator`, implement `_preprocess_input`, `_fit`, and
    `_generate_description`, and return `MetaResult` or a specialized result subclass via
    `_make_result`.
- Result objects store maps, tables, metadata, descriptions, and fitted estimator state. Specialized
    results, such as `CBMRResult`, add workflow-specific helpers while preserving `MetaResult`
    behavior.
- Reusable preprocessing, validation, normalization, and numerical-stability logic is factored into
    small private helpers near the top of a module or near the class that uses it.
- Optional dependencies are isolated. Modules requiring torch raise explicit `ImportError`s with
    installation guidance, and public package imports defer optional modules where possible.
- Computational backends are selected with explicit options and validation, not implicit fallback
    behavior.

### Imports and dependency organization

Imports are grouped as standard library, third-party packages, then local `nimare` imports. Prefer importing concrete helpers from the package module that owns them, and keep optional imports guarded when importing them eagerly would make unrelated APIs
unavailable.

Dependencies are declared in `setup.cfg`, with optional extras such as `cbmr` for torch-backed CBMR
functionality. Test, documentation, and optional runtime dependencies are separated in extras.

### Error handling, logging, configuration, and testing

- Validate public options early and raise `ValueError` or `TypeError` with precise, actionable
    messages.
- Use module loggers (`LGR = logging.getLogger(__name__)`) and parameterized logging calls for
    runtime messages.
- Do not silently ignore invalid inputs. Either raise an error or log a warning consistent with the
    rest of the module.
- Configuration lives in `pyproject.toml` and `setup.cfg`.
- Tests use pytest fixtures, `pytest.mark.parametrize`, deterministic random states, and explicit
    skip conditions for optional dependencies such as torch and CUDA.

### Formatting and documentation style

- Code is formatted with Black and imports with isort.
- Public classes, methods, and functions use NumPy-style docstrings with sections such as
    `Parameters`, `Returns`, `Attributes`, and `Notes` when applicable.
- Module docstrings are concise sentence-case summaries.
- Comments should explain non-obvious reasoning, not restate straightforward code.
- Descriptive strings and generated method descriptions should be clear prose and preserve citation
    formatting where used.

## Recognizing contributions

We welcome and recognize all contributions from documentation to testing to code development.
You can see a list of current contributors in our [zenodo][link_zenodo] file.
If you are new to the project, don't forget to add your name and affiliation there!

## Thank you!

You're awesome.

.. note::
    These guidelines are based on contributing guidelines from the [STEMMRoleModels][link_stemmrolemodels] project.

[link_github]: https://github.com/
[link_nimare]: https://github.com/neurostuff/NiMARE
[link_signupinstructions]: https://help.github.com/articles/signing-up-for-a-new-github-account
[link_react]: https://github.com/blog/2119-add-reactions-to-pull-requests-issues-and-comments
[link_issues]: https://github.com/neurostuff/NiMARE/issues
[link_labels]: https://github.com/neurostuff/NiMARE/labels
[link_discussingissues]: https://help.github.com/articles/discussing-projects-in-issues-and-pull-requests
[link_neurostars]: https://neurostars.org
[link_neurostars_nimare]: https://neurostars.org/tags/c/software-support/234/nimare

[link_pullrequest]: https://help.github.com/articles/creating-a-pull-request/
[link_fork]: https://help.github.com/articles/fork-a-repo/
[link_pushpullblog]: https://www.igvita.com/2011/12/19/dont-push-your-pull-requests/
[link_branches]: https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/
[link_updateupstreamwiki]: https://help.github.com/articles/syncing-a-fork/
[link_stemmrolemodels]: https://github.com/KirstieJane/STEMMRoleModels
[link_zenodo]: https://github.com/neurostuff/NiMARE/blob/main/.zenodo.json
