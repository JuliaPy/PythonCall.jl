# Repository Instructions

## Running tests
- **Julia tests**: Run from the project root with `julia --project -e 'using Pkg; Pkg.test()'`. Expect a warning about the General registry being unreachable in locked-down environments; the suite still finishes (PyCall tests are marked broken/Skipped).
- **Python tests**:
  - Copy `pysrc/juliacall/juliapkg-dev.json` to `pysrc/juliacall/juliapkg.json` before running (do **not** commit this copy).
  - Execute with `uv run pytest -s --nbval ./pytest` (add `--cov=pysrc` when coverage is needed).
  - Sometimes `juliapkg` requires Julia 1.10–1.11; `juliaup` already provides 1.11.7 in this environment.

The majority of tests live in the Julia package; Python tests cover functionality that cannot be exercised from Julia (e.g., JuliaCall-specific behavior). Run both suites—typically Julia first—in whichever order makes sense.

## Meta instructions
- When you discover environment quirks, false assumptions, process fixes, or any other generally useful info, update this AGENTS.md so future coding agents have the information.
