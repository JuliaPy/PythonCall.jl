# Getting Started

## You will need

* Julia 1.0 or higher.
* Python 3.5 or higher.

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
- `PYTHONJL_EXE`: Path to the Python executable. Or the special value `CONDA` which uses
  Python from the default conda environment, or `CONDA:{env}` to use the given environment.
- `PYTHONJL_LIB`: Path to the Python library. Normally this is inferred from the Python
  executable, but can be over-ridden.
- `JULIAPY_EXE`: Path to the Julia executable.
- `JULIAPY_LIB`: Path to the Julia library. Normally this is inferred from the Julia
  executable, but can be over-ridden.
