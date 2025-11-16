# Repository Instructions

## Running tests
- **Julia tests**: Run from the project root with `julia -e 'using Pkg; Pkg.test()'`.
- **Python tests**:
  - Copy `juliapkg-dev.json` to `juliapkg.json` before running (do **not** commit this copy).
  - Execute with `uv run pytest -s --nbval --cov=pysrc ./pytest`. Skip `--cov` unless coverage is required.

The majority of tests live in the Julia package; Python tests cover functionality that cannot be exercised from Julia (e.g., JuliaCall-specific behavior). Run both suites—typically Julia first—in whichever order makes sense.
