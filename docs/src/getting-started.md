# Getting Started

## You will need

* Julia 1.0 or higher — download [here](https://julialang.org/downloads).
* Python 3.5 or higher — download [here](https://www.python.org/downloads) or set `JULIA_PYTHONCALL_EXE=CONDA` (see below).

## Install the Julia package `PythonCall`

```julia
using Pkg
pkg"add PythonCall"
```

## Install the Python package `juliacall` (optional)

This step is only required if you wish to call Julia from Python.

Currently `juliacall` is shipped with the source of the Julia package, and must be
pip-installed manually. The following should work in most shells (including PowerShell):

```bash
pip install $(julia -e "using PythonCall; print(PythonCall.juliacall_pipdir)")
```

Alternatively you can just copy the package (at `PythonCall.juliacall_dir`) to somewhere in your PYTHONPATH.

Note that this is a [very small](https://github.com/cjdoris/PythonCall.jl/blob/master/juliacall/__init__.py)
"bootstrap" package whose sole job is to locate and load Julia; the main functionality is in
the main Julia package. Hence it is not necessary to upgrage `juliacall` every time
you upgrade `PythonCall`.

Note also that regardless of installing `juliacall`, a module called `juliacall` will
always be loaded into the interpreter by `PythonCall`. This means that other Python
packages can always `import juliacall`.

## Environment variables

If Julia and Python are in your PATH, then no further set-up is required.
Otherwise, the following environment variables control how the package finds these.
- `JULIA_PYTHONCALL_EXE`: Path to the Python executable. Or the special value `CONDA` which uses
  Python from the default conda environment, or `CONDA:{env}` to use the given environment.
  In this case, if `conda` is not detected then `Conda.jl` will automatically install
  [`miniconda`](https://docs.conda.io/en/latest/miniconda.html) in your Julia depot.
- `JULIA_PYTHONCALL_LIB`: Path to the Python library. Normally this is inferred from the Python
  executable, but can be over-ridden.
- `PYTHON_JULIACALL_EXE`: Path to the Julia executable.
- `PYTHON_JULIACALL_LIB`: Path to the Julia library. Normally this is inferred from the Julia
  executable, but can be over-ridden.
