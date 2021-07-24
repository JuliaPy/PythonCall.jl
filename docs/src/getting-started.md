# Getting Started

## Install the Julia package `PythonCall`

```julia
julia> using Pkg

pkg> add PythonCall
```

## Install the Python package `juliacall` (optional)

This step is only required if you wish to call Julia from Python.

Currently `juliacall` is shipped with the source of the Julia package, and must be
pip-installed manually. The following should work in most shells (including PowerShell):

```bash
pip install $(julia -e ":PythonCall|>string|>Base.find_package|>dirname|>dirname|>print")
```

Alternatively you can just copy the package from the `PythonCall` source directory to somewhere in your PYTHONPATH.

Note that this is a [very small](https://github.com/cjdoris/PythonCall.jl/blob/master/juliacall/__init__.py)
"bootstrap" package whose sole job is to locate and load Julia; the main functionality is in
the main Julia package. Hence it is not necessary to upgrage `juliacall` every time
you upgrade `PythonCall`.

Note also that regardless of installing `juliacall`, a module called `juliacall` will
always be loaded into the interpreter by `PythonCall`. This means that other Python
packages can always `import juliacall`.

## Environment variables

- `JULIA_PYTHONCALL_EXE`: By default, `PythonCall` manages its own installation of Python
  specific to a particular Julia environment, so that the set of installed Python packages
  is isolated between environments. To instead use a pre-installed version of Python, set
  this variable to its path. It can simply be set to `python` if it is in your `PATH`.
- `PYTHON_JULIACALL_EXE`: The path to the Julia executable. By default, it uses `julia`.
