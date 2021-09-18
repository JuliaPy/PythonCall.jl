# The Python module *juliacall*

## Installation

In the future, the package will be available on PyPI and conda.
For now, you can pip install this package directly from github as follows:

```bash
pip install git+https://github.com/cjdoris/PythonCall.jl
```

Developers may wish to clone the repo directly and pip install the module in editable mode.
This guarantees you are using the latest version of PythonCall in conjunction with juliacall.

Note also that regardless of installing `juliacall`, a module called `juliacall` will
always be loaded into the interpreter by `PythonCall`. This means that other Python
packages can always `import juliacall`.

## Getting started

For interactive or scripting use, the simplest way to get started is:

```python
from juliacall import Main as jl
```

This loads a single variable `jl` (a [`ModuleValue`](#juliacall.ModuleValue)) which represents the `Main` module in Julia, from which all of Julia's functionality is available.

If you are writing a package which uses Julia, then to avoid polluting the global `Main` namespace you should do:

```python
import juliacall; jl = juliacall.newmodule("SomeName");
```

Now you can do `jl.rand(jl.Bool, 5, 5)`, which is equivalent to `rand(Bool, 5, 5)` in Julia.

When a Python value is passed to Julia, then typically it will be converted according to [this table](@ref py2jl) with `T=Any`.
Sometimes a more specific type will be used, such as when assigning to an array whose element type is known.

When a Julia value is returned to Python, it will normally be converted according to [this table](@ref jl2py).

## Managing Julia dependencies

juliacall manages its Julia dependencies using [Pkg](https://pkgdocs.julialang.org/v1) for
packages and [jill](https://pypi.org/project/jill/) for Julia itself.
If a suitable version of julia is not found on your system, it will automatically be
downloaded and installed into `~/.julia/pythoncall`.
A Julia environment is automatically created when juliacall is loaded, is activated, and is
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
            "compat": "0.5",
            "url": "http://github.com/JuliaLang/Example.jl",
            "path": "/path/to/the/package",
            "rev": "master",
            "dev": false, // when true, uses Pkg.dev not Pkg.add
        }
    }
}
```
All parts are optional, except that the UUID of each package is required.

When juliacall starts, it will ensure the latest compatible version of julia is installed,
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
