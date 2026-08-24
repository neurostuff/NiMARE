# NiMARE development conventions

This guide summarizes the software development conventions and hierarchical code design patterns
used in this repository. Use it as a reference when adding or reviewing code.

## Code organization and module hierarchy

NiMARE is organized as a domain-focused Python package under `nimare/`.

- Core infrastructure lives in package-level modules such as `nimare/base.py`,
  `nimare/estimator.py`, `nimare/results.py`, `nimare/utils.py`, and `nimare/transforms.py`.
- Analysis methods are grouped by domain. For example, `nimare/meta/` contains meta-analytic
  estimators, with coordinate-based methods in `nimare/meta/cbma/`, image-based methods in
  `nimare/meta/ibma.py`, kernels in `nimare/meta/kernel.py`, CBMR in `nimare/meta/cbmr.py`, and
  CBMR torch models in `nimare/meta/models.py`.
- Public APIs are exposed from package `__init__.py` files. Optional CBMR exports are lazily
  imported in `nimare/meta/__init__.py` so importing `nimare.meta` does not require torch.
- Tests live in `nimare/tests/` and are grouped by feature or module, for example
  `test_meta_ale.py`, `test_meta_ibma.py`, and `test_meta_cbmr.py`.
- Examples and benchmarks are kept outside the package in `examples/` and `benchmarks/`.

## Naming conventions

- Modules and files use lowercase, descriptive names with underscores when needed
  (`cbmr.py`, `models.py`, `test_meta_cbmr.py`).
- Classes use `PascalCase` (`CBMREstimator`, `CBMRInference`, `CBMRResult`).
- Functions, methods, variables, and pytest fixtures use `snake_case`.
- Private helpers and implementation details are prefixed with an underscore
  (`_validate_incidence_threshold`, `_build_group_foci_matrices`).
- Constants use uppercase names (`DEFAULT_GROUP_NAME`, `DEFAULT_FLOAT_DTYPE`).
- Tests are named `test_<behavior>` and should describe the behavior under test rather than the
  implementation detail alone.

## Design patterns and abstractions

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

## Imports and dependency organization

Imports are grouped as standard library, third-party packages, then local `nimare` imports. isort is
configured with the Black profile. Prefer importing concrete helpers from the package module that
owns them, and keep optional imports guarded when importing them eagerly would make unrelated APIs
unavailable.

Dependencies are declared in `setup.cfg`, with optional extras such as `cbmr` for torch-backed CBMR
functionality. Test, documentation, and optional runtime dependencies are separated in extras.

## Error handling, logging, configuration, and testing

- Validate public options early and raise `ValueError` or `TypeError` with precise, actionable
  messages.
- Use module loggers (`LGR = logging.getLogger(__name__)`) and parameterized logging calls for
  runtime messages.
- Do not silently ignore invalid inputs. Either raise an error or log a warning consistent with the
  rest of the module.
- Configuration lives in `pyproject.toml` and `setup.cfg`. Black and flake8 use a 99-character line
  length, and docstrings follow the NumPy convention.
- Tests use pytest fixtures, `pytest.mark.parametrize`, deterministic random states, and explicit
  skip conditions for optional dependencies such as torch and CUDA.
- Numerical tests use `numpy.testing` or `pandas.testing` helpers instead of hand-written array or
  DataFrame comparisons.

## Formatting and documentation style

- Code is formatted with Black and imports with isort.
- Public classes, methods, and functions use NumPy-style docstrings with sections such as
  `Parameters`, `Returns`, `Attributes`, and `Notes` when applicable.
- Module docstrings are concise sentence-case summaries.
- Comments should explain non-obvious reasoning, not restate straightforward code.
- Descriptive strings and generated method descriptions should be clear prose and preserve citation
  formatting where used.
