# Repository Instructions

## Running tests
- **Julia tests**: Run from the project root with `julia --project -e 'using Pkg; Pkg.test()'`. Expect a warning about the General registry being unreachable in locked-down environments; PyCall tests are marked broken/skipped. With the currently installed Julia 1.13.0, the functional tests pass but Aqua's persistent-task subprocess check may fail because its `done.log` is not created.
- **Python tests**:
  - Copy `pysrc/juliacall/juliapkg-dev.json` to `pysrc/juliacall/juliapkg.json` before running (do **not** commit this copy).
  - Execute with `uv run pytest -s --nbval ./pytest` (add `--cov=pysrc` when coverage is needed).
- Sometimes `juliapkg` requires Julia 1.10–1.11. Check `juliaup status` rather than
  assuming 1.11.7 is installed: as of 2026-10-10 this environment only has Julia
  1.13 builds. JuliaPkg selecting Julia 1.13.1 under uv's downloaded Python 3.10
  fails to load `libjulia-codegen` because the Python process has loaded a
  `libstdc++.so.6` that lacks `GLIBCXX_3.4.30`.
- `julia --project=docs docs/make.jl` requires a valid Git `origin` so Documenter can infer
  source links; it fails during `makedocs` in checkouts without one.

The majority of tests live in the Julia package; Python tests cover functionality that cannot be exercised from Julia (e.g., JuliaCall-specific behavior). Run both suites—typically Julia first—in whichever order makes sense.

## Meta instructions
- When you discover environment quirks, false assumptions, process fixes, or any other generally useful info, update this AGENTS.md so future coding agents have the information.
