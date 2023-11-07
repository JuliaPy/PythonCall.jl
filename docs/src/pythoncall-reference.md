# [PythonCall API Reference](@id py-reference)

## `Py` objects

```@docs
Py
pybuiltins
```

## Constructors

These functions construct Python objects of builtin types from Julia values.

```@docs
pybool
pycollist
pybytes
pycomplex
pydict
pyfloat
pyfrozenset
pyint
pylist
pyrange
pyrowlist
pyset
pyslice
pystr
pytuple
```

## Builtins

These functions mimic the Python builtin functions or keywords of the same name.

```@docs
pyall
pyany
pyascii
pycall
pycallable
pycompile
pycontains
pydelattr
pydelitem
pydir
pyeval
@pyeval
pyexec
@pyexec
pygetattr
pygetitem
pyhasattr
pyhasitem
pyhash
pyhelp
pyimport
pyin
pyis
pyisinstance
pyissubclass
pyiter
pylen
pynext
pyprint
pyrepr
pysetattr
pysetitem
pytype(::Any)
pywith
```

## Conversion to Julia

These functions convert Python values to Julia values, using the rules documented [here](@ref jl2py).

```@docs
pyconvert
@pyconvert
```

## Wrap Julia values

These functions explicitly wrap Julia values into Python objects, documented [here](@ref julia-wrappers).

As documented [here](@ref py2jl), Julia values are wrapped like this automatically on
conversion to Python, unless the value is immutable and has a corresponding Python type.

```@docs
pyjl
pyjlraw
pyisjl
pyjlvalue
pybinaryio
pytextio
```

## Arithmetic

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

## Logic

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

## Create classes

These functions can be used to create new Python classes where the functions are implemented
in Julia. You can instead use [`@pyeval`](@ref) etc. to create pure-Python classes.

```@docs
pytype(::Any, ::Any, ::Any)
pyfunc
pyclassmethod
pystaticmethod
pyproperty
```

## [Wrapper types](@id python-wrappers)

The following types wrap a Python object, giving it the semantics of a Julia object. For example `PyList(x)` interprets the Python sequence `x` as a Julia abstract vector.

Apart from a few fundamental immutable types, conversion from Python to Julia `Any` will return a wrapper type such as one of these, or simply `Py` if no wrapper type is suitable.

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
PyException
```

### [Custom wrappers](@id python-wrappers-custom)

Here is a minimal example of defining a wrapper type. You may add methods, fields and a
supertype to the type to specialise its behaviour. See any of the above wrapper types for
examples.

```julia
# The new type with a field for the Python object being wrapped.
struct MyType
    py::Py
end

# Says that the object is a wrapper.
ispy(x::MyType) = true

# Says how to access the underlying Python object.
Py(x::MyType) = x.py
```

## `@py` and `@pyconst`

```@docs
@py
@pyconst
```

## The Python interpreter

These functions are not exported. They give information about which Python interpreter is
being used.
```@docs
PythonCall.python_version
PythonCall.python_executable_path
PythonCall.python_library_path
PythonCall.python_library_handle
```

## Low-level API

The functions here are not exported. They are mostly unsafe in the sense that you can
crash Julia by using them incorrectly.

```@docs
PythonCall.pynew
PythonCall.pyisnew
PythonCall.pycopy!
PythonCall.pydel!
PythonCall.unsafe_pynext
```
