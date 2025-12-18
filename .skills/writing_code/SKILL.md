---
name: writing code
description: Write code to implement features or fix issues.
---

# Writing Code

- Check for backwards compatibility when modifying public functions, classes, or CLI behavior, especially if they existed before the current branch.
- Prefer test-driven development when feasible: write or update tests first, then implement code to pass those tests.
- Prefer reusing or extending existing functions and patterns over introducing new APIs; search the codebase for similar functionality before adding new modules.
- Follow NiMARE’s coding style:
  - Adhere to PEP8 and use the same scikit-learn–like estimator pattern (`fit`, `transform`, etc.) for user-facing classes.
  - Write docstrings in numpydoc style for public objects and keep parameter names consistent between docs and code.
- Keep changes focused and small; group logically related changes into a single PR and avoid mixing unrelated refactors with feature or bugfix work.
- After implementing changes, run the relevant tests (see the “running tests” skill) before opening or updating a pull request.
