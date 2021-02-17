# Getting Started

## You will need

* Julia 1.0 or higher.
* Python 3.5 or higher.

## Install the Julia package

```julia
using Pkg
pkg"add https://github.com/cjdoris/Python.jl"
```

## Install the Python package (optional)

This step is only required if you wish to call Julia from Python.

Currently the Python package is shipped with the source of the Julia package, and must be
pip-installed manually. The following should work in most shells (including PowerShell):

```bash
pip install --upgrade $(julia -e "using Python; print(Python.juliapypath)")
```

Note that this is a very small "bootstrap" package whose sole job is to locate and load
Julia; the main functionality is in the main Julia package. Hence it is not necessary to
upgrage the Python package every time you upgrade the Julia one.

## Environment variables

If Julia and Python are in your PATH, then no further set-up is required.
Otherwise, the following environment variables control how the package finds these.
- `PYTHONJL_EXE`: Path to the Python executable. Or the special value `CONDA` which uses
  Python from the default conda environment, or `CONDA:{env}` to use the given environment.
- `PYTHONJL_LIB`: Path to the Python library. Normally this is inferred from the Python
  executable, but can be over-ridden.
- `JULIAPY_EXE`: Path to the Julia executable.
- `JULIAPY_LIB`: Path to the Julia library. Normally this is inferred from the Julia
  executable, but can be over-ridden.
