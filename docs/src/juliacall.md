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
