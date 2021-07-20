# The Julia module `PythonCall`

## `Py`

```@docs
Py
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

### Construct Python objects

These functions convert Julia values into Python objects of standard types.

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

### Wrap Julia values

These functions wrap Julia values into Python objects, documented [here](@ref julia-wrappers).

```@docs
pyjl
pyjlraw
pyisjl
pyjlvalue
pytextio
pybinaryio
```

### Python builtins

```@docs
pyconvert
@pyconvert
pyimport
pyimport_conda
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
pytruth
pyissubclass
pyisinstance
pyhash
pyiter
```

### Numeric functions

```@docs
pyneg
pypos
pyabs
pyinv
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

### Comparisons

```@docs
pyeq
pyne
pyle
pylt
pyge
pygt
```

## [Wrapper types](@id python-wrappers)

```@docs
PyList
PySet
PyDict
PyIterable
PyArray
PyIO
PyTable
PyPandasDataFrame
PyObjectArray
```
