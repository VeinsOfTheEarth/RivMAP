# Repository history

This repository began as a GitHub-oriented packaging of the original RivMAP
MATLAB toolbox. That initial reorganization was intentionally minimal: MATLAB
sources moved to `src/`, the demo to `examples/`, data to `data/`, and the
walkthrough to `docs/`; the demo was changed only to use repository-relative
paths.

RivMAPy is the subsequent Python port. Its implementation lives in
`src/rivmapy/`, with Python demonstrations under `examples/` and regression
tests under `tests/`. The port preserves the original scientific intent while
making several algorithms deterministic, bounded, and explicit; these are
substantive implementation changes rather than packaging-only changes. See
`legacy_behavior.md` for the precise compatibility choices and corrected
legacy defects.

The MATLAB source, demo, data, and walkthrough remain in place as provenance
and are not modified by the Python workflows.
