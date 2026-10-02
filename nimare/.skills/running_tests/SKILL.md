---
name: running tests
description: Run tests to ensure code functionality and correctness.
---

# Running Tests

- DO NOT RUN ALL TESTS AT ONCE UNLESS EXPLICITLY INSTRUCTED TO DO SO.
- Prefer running only the tests relevant to the code you have modified or added.
- Use `pytest` to run tests from the repository root, for example:

  ```bash
  pytest nimare/tests/test_annotate_gclda.py::test_gclda_symmetric
  ```

- For a slightly broader check without performance-heavy tests, mirror the `Makefile` behavior:

  ```bash
  pytest -m "not performance_estimators and not performance_correctors and not performance_smoke and not cbmr_importerror" --cov=nimare nimare
  ```

- Run tests before and after significant refactors or API changes to confirm behavior is preserved.
