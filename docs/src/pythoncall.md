# The Julia module *PythonCall*

## Installation

This package is in the general registry, so to install just type `]` in the Julia REPL and run:

```julia-repl
pkg> add PythonCall
```

## Getting started

Import the module with:

```julia-repl
julia> using PythonCall
```

By default this will initialize a conda environment in your Julia environment, install
Python into it, load the corresponding Python library and initialize an interpreter.

Now you can interact with Python as follows:

```julia-repl
julia> re = pyimport("re")
Python module: <module 're' from '[...]/lib/re.py'>

julia> words = re.findall("[a-zA-Z]+", "PythonCall.jl is very useful!")
Python list: ['PythonCall', 'jl', 'is', 'very', 'useful']

julia> sentence = Py(" ").join(words)
Python str: 'PythonCall jl is very useful'

julia> pyconvert(String, sentence)
"PythonCall jl is very useful"
```

In this example:
- We used [`pyimport`](@ref) to import the `re` module. Equivalently we could have done
  `@py import re` (see [`@py`](@ref)).
- We called its `findall` function on a pair of strings, which were automatically
  converted to Python strings (see [Conversion to Python](@ref jl2py)).
- We called [`Py`](@ref) to explicitly convert a string to a Python string, so that we
  could call its `join` method. All Python objects are of type `Py`.
- We called [`pyconvert`](@ref) to convert the Python string `sentence` to a Julia string
  (see [Conversion to Julia](@ref py2jl)).

Read on to find out what else you can do.

## `Py`

```@docs
Py
@pyconst
```

The object `pybuiltins` has all the standard Python builtin objects as its properties.
Hence you can access `pybuiltins.None` and `pybuiltins.TypeError`.

## `@py`

```@docs
@py
```

## Python functions

Most of the functions in this section are essentially Python builtins with a `py` prefix.
For example `pyint(x)` converts `x` to a Python `int` and is equivalent to `int(x)` in
Python when `x` is a Python object.

Notable exceptions are:
- [`pyconvert`](@ref) to convert a Python object to a Julia object.
- [`pyimport`](@ref) to import a Python module.
- [`pyjl`](@ref) to directly wrap a Julia object as a Python object.
- [`pyclass`](@ref) to construct a new class.
- [`pywith`](@ref) to emulate the Python `with` statement.

If a Julia value is passed as an argument to one of these functions, it is converted to a
Python value using the rules documented [here](@ref jl2py).

### Constructors

These functions construct Python objects of builtin types from Julia values.

```@docs
pybool
pyint
pyfloat
pycomplex
pystr
pybytes
pytuple
pylist
pycollist
pyrowlist
pyset
pyfrozenset
pydict
pyslice
pyrange
pymethod
pytype
pyclass
```

### Builtins

These functions mimic the Python builtin functions or keywords of the same name.

```@docs
pyimport
pywith
pyis
pyrepr
pyascii
pyhasattr
pygetattr
pysetattr
pydelattr
pydir
pycall
pylen
pycontains
pyin
pygetitem
pysetitem
pydelitem
pyissubclass
pyisinstance
pyhash
pyiter
pynext
pyhelp
pyprint
pyall
pyany
pycallable
```

### Conversion to Julia

These functions convert Python values to Julia values, using the rules documented [here](@ref jl2py).

```@docs
pyconvert
@pyconvert
```

### Wrap Julia values

These functions explicitly wrap Julia values into Python objects, documented [here](@ref julia-wrappers).

As documented [here](@ref py2jl), Julia values are wrapped like this automatically on
conversion to Python, unless the value is immutable and has a corresponding Python type.

```@docs
pyjl
pyjlraw
pyisjl
pyjlvalue
pytextio
pybinaryio
```

### Arithmetic

These functions are equivalent to the corresponding Python arithmetic operators.

Note that the equivalent Julia operators are overloaded to call these when all arguments
are `Py` (or `Number`). Hence the following are equivalent: `Py(1)+Py(2)`, `Py(1)+2`,
`pyadd(1, 2)`, `pyadd(Py(1), Py(2))`, etc.

```@docs
pyneg
pypos
pyabs
pyinv
pyindex
pyadd
pysub
pymul
pymatmul
pypow
pyfloordiv
pytruediv
pymod
pydivmod
pylshift
pyrshift
pyand
pyxor
pyor
pyiadd
pyisub
pyimul
pyimatmul
pyipow
pyifloordiv
pyitruediv
pyimod
pyilshift
pyirshift
pyiand
pyixor
pyior
```

### Logic

These functions are equivalent to the corresponding Python logical operators.

Note that the equivalent Julia operators are overloaded to call these when all arguments
are `Py` (or `Number`). Hence the following are equivalent: `Py(1) < Py(2)`, `Py(1) < 2`,
`pylt(1, 2)`, `pylt(Py(1), Py(2))`, etc.

Note that the binary operators by default return `Py` (not `Bool`) since comparisons in
Python do not necessarily return `bool`.

```@docs
pytruth
pynot
pyeq
pyne
pyle
pylt
pyge
pygt
```

## Managing Python dependencies

PythonCall manages its Python dependencies using Conda. A Conda environment is automatically
created in your active Julia environment when PythonCall is loaded, is initialised with
at least `python` and `pip`, and is activated.

If your project requires more Python dependencies, use the mechanisms below to ensure they
are automatically installed.

We **strongly recommend that you specify Conda dependencies** if possible, instead of pip
or script dependencies. This is because Conda can account for all inter-dependencies between
packages and so prevent incompatible combinations of packages from being installed.

### PythonCallDeps.toml

If you put a file called `PythonCallDeps.toml` in a project/package/environment which
depends on PythonCall, then the dependencies therein will be automatically installed into
the Conda environment.

Here is an example (all parts are optional):
```toml
[conda]
packages = ["python>=3.6", "scikit-learn"]
channels = ["conda-forge"]

[pip]
packages = ["numpy>=1.21"]
# indexes = [...]

[script]
# expr = "some_julia_expression()"
# file = "/path/to/julia/script.jl"
```

When PythonCall starts, it will ensure the Conda environment has the given Conda and pip
packages installed, and will run the script if specified.

### The Deps submodule

Instead of manually editing `PythonCallDeps.toml`, you can use the submodule
`PythonCall.Deps` to manage the Python dependencies of the current Julia project.

```@docs
PythonCall.Deps.status
PythonCall.Deps.add
PythonCall.Deps.rm
PythonCall.Deps.resolve
PythonCall.Deps.conda_env
PythonCall.Deps.user_deps_file
```

### The Python interpreter

By default, `python` is automatically installed into the Conda environment mentioned above.

To use a different interpreter, you can set the environment variable `JULIA_PYTHONCALL_EXE`
to its path before importing PythonCall. You can set it to `python` if it is in your PATH.

You can also set it to the special value `"@PyCall"` which will use the same interpreter as
PyCall.

Note that using a non-default interpreter will disable all dependency management: no Conda
environment will be created and no packages will be automatically installed. It is up to the
user to ensure any required packages are installed.

## Writing packages which depend on *PythonCall*

### Example

See [https://github.com/cjdoris/Faiss.jl] for an example package which wraps the Python FAISS
package.

### Precompilation

You may not interact with Python during module precompilation. Therefore, instead of
```julia
module MyModule
  using PythonCall
  const foo = pyimport("foo")
  bar() = foo.bar() # will crash when called
end
```
you must do
```julia
module MyModule
  using PythonCall
  const foo = PythonCall.pynew() # initially NULL
  function __init__()
    PythonCall.pycopy!(foo, pyimport("foo"))
  end
  bar() = foo.bar() # now ok
end
```

### Dependencies

If your package depends on some Python packages, you must write a `PythonCallDeps.toml` file.
See [Managing Python dependencies](@ref).

## Low-level API

The functions here are not exported. They are mostly unsafe in the sense that you can
crash Julia by using them incorrectly.

```@docs
PythonCall.pynew
PythonCall.pyisnull
PythonCall.pycopy!
PythonCall.getptr
PythonCall.pydel!
PythonCall.unsafe_pynext
```
