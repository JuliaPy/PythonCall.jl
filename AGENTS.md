# Repository Instructions

## Running tests
- **Julia tests**: Run from the project root with `julia --project=@. -e 'using Pkg; Pkg.test()'`. Running without `--project=@.` fails because the global environment lacks the package name/UUID. Expect a warning about the General registry being unreachable in locked-down environments; the suite still finishes (PyCall tests are marked broken/Skipped).
- **Python tests**:
  - Copy `pysrc/juliacall/juliapkg-dev.json` to `pysrc/juliacall/juliapkg.json` before running (do **not** commit this copy).
  - Execute with `uv run pytest -s --nbval ./pytest` (add `--cov=pysrc` only when coverage is needed).
  - These tests currently require Julia 1.10–1.11. They failed here because `juliapkg` could not download metadata/install Julia through the proxy (tunnel 403) and the installed Julia 1.12.1 is outside the requested range. Without internet/proxy access to julialang.org, expect the suite to fail early while resolving Julia versions.

The majority of tests live in the Julia package; Python tests cover functionality that cannot be exercised from Julia (e.g., JuliaCall-specific behavior). Run both suites—typically Julia first—in whichever order makes sense.

## Meta instructions
- When you discover environment quirks, false assumptions, or process fixes, add/update this AGENTS.md so future coding agents have the information.
