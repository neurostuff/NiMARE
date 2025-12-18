---
name: reviewing code
description: how to review code effectively
---

# Reviewing Code Effectively
- Start by understanding the motivation for the change: read the linked issue, PR description, and any relevant design discussion.
- Use `git diff` to examine the scope of changes and verify that only necessary files and lines were modified.
- Check for consistency with NiMARE’s style and patterns:
  - PEP8 compliance and clear, numpydoc-style docstrings for public APIs.
  - Use of the scikit-learn–like estimator pattern where appropriate.
- Confirm that new or modified functionality is covered by targeted tests, and that tests are focused and efficient.
- When appropriate, run the relevant tests locally (not the entire test suite) to validate critical behavior or bugfixes.
- Ensure documentation and examples are updated when APIs or behavior change (e.g., `docs/dev_guide.rst`, `docs/api.rst`, or relevant method docs).
- Offer specific, constructive feedback that suggests concrete improvements or clarifications, and be explicit about what is blocking versus optional.
