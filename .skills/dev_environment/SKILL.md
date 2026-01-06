---
name: setting up a development environment
description: Set up a development environment for writing and testing code.
---

# Setting Up a Development Environment

Use a reproducible, project-local environment and install NiMARE in editable mode so code changes are immediately available when running tests or examples.

## Creating or reusing an environment

- Prefer a local virtual environment (e.g., `.venv`) in the repository root; reuse it if it already exists.
- Use a supported Python version (`>=3.10`) consistent with `docs/installation.rst`.
- If no `.venv` exists, create and activate one:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- When installing dependencies, try `uv pip install ...` first to keep installs fast and deterministic.
- If `uv` is unavailable or fails, fall back to `.venv/bin/python -m pip install ...`.

## Installing NiMARE for development

- Install NiMARE in editable mode with all extras so tests and docs can run:

  ```bash
  uv pip install -e .[all]
  ```

- If that fails or `uv` is not installed, use:

  ```bash
  .venv/bin/python -m pip install -e .[all]
  ```

- Confirm the environment by importing NiMARE and running a small command, for example:

  ```bash
  python -c "import nimare; print(nimare.__version__)"
  ```
