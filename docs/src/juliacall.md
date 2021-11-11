# The Python module JuliaCall

## Installation

It's as simple as
```bash
pip install juliacall
```

Developers may wish to clone the repo (https://github.com/cjdoris/PythonCall.jl) directly
and pip install the module in editable mode. This guarantees you are using the latest
version of PythonCall in conjunction with JuliaCall.

## Getting started

For interactive or scripting use, the simplest way to get started is:
```python
from juliacall import Main as jl
```

This loads a single variable `jl` which represents the `Main` module in Julia,
from which all of Julia's functionality is available:
```python
jl.println("Hello from Julia!")
# Hello from Julia!
x = jl.rand(range(10), 3, 5)
x._jl_display()
# 3×5 Matrix{Int64}:
#  8  1  7  0  6
#  9  2  1  4  0
#  1  8  5  4  0
import numpy
numpy.sum(x, axis=0)
# array([18, 11, 13,  8,  6], dtype=int64)
```

In this example:
- We called the `jl.println` function to print a message.
- We called the `jl.rand` function to generate an array of random integers. Note that the
  first argument is `range(10)` which is converted to `0:9` in Julia.
- We called its special `_jl_display()` to show it using Julia's display mechanism.
- We called the `numpy.sum` function to sum each column of `x`. This automatically converted
  `x` to a NumPy array. (We could have done `jl.sum(x, dims=1)` too.)

If you are writing a package which uses Julia, then to avoid polluting the global `Main`
namespace you instead should start with:
```python
import juliacall; jl = juliacall.newmodule("SomeName");
```

What to read next:
- The main functionality of this package is in `AnyValue` objects, which represent Julia
  objects, [documented here](@ref julia-wrappers).
- If you need to install Julia packages, [read here](@ref julia-deps).
- When you call a Julia function, such as `jl.rand(...)` in the above example, its
  arguments are converted to Julia according to [this table](@ref py2jl-conversion) and
  its return value is converted to Python according to [this table](@ref jl2py-conversion).

## [Managing Julia dependencies](@id julia-deps)

JuliaCall manages its Julia dependencies using [Pkg](https://pkgdocs.julialang.org/v1).
If a suitable version of Julia is not found on your system, it will automatically be
downloaded and installed into `~/.julia/pythoncall/julia-VERSION`.
A Julia environment is automatically created when JuliaCall is loaded, is activated, and is
initialised with at least PythonCall. If you are using a virtual or conda environment then
the Julia environment is created there, otherwise a global environment is created at
`~/.julia/environments/PythonCall`.

If your project requires more Julia dependencies, use the mechanisms below to ensure they
are automatically installed.

### juliacalldeps.json

If you put a file called `juliacalldeps.json` in a Python package, then the dependencies
therein will be automatically installed into the Julia environment.

Here is an example:
```json
{
    "julia": "1.5",
    "packages": {
        "Example": {
            "uuid": "7876af07-990d-54b4-ab0e-23690620f79a",
            "version": "0.5, 0.6",
            "url": "http://github.com/JuliaLang/Example.jl",
            "path": "/path/to/the/package",
            "rev": "master",
            "dev": false, // when true, uses Pkg.dev not Pkg.add
        }
    }
}
```
All parts are optional, except that the UUID of each package is required. Typically you
will just include the UUID and version fields.

When JuliaCall starts, it will ensure the latest compatible version of Julia is installed,
and will ensure the given packages are installed.

## Utilities

`````@customdoc
juliacall.newmodule - Function

```python
newmodule(name)
```

A new module with the given name.
`````

`````@customdoc
juliacall.As - Class

```python
As(x, T)
```

When passed as an argument to a Julia function, is interpreted as `x` converted to Julia type `T`.
`````
