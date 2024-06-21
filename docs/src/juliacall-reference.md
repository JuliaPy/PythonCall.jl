# JuliaCall API Reference

## Constants

`````@customdoc
juliacall.Main - Constant

The Julia `Main` module, as a [`Jl`](#juliacall.Jl).

In interactive scripts, you can use this as the main entry-point to JuliaCall:
```python
from juliacall import Main as jl
jl.println("Hello, world!")
```

In packages, use [`newmodule`](#juliacall.newmodule) instead.
`````

The modules `Base`, `Core` and `PythonCall` are also available.

## Utilities

`````@customdoc
juliacall.newmodule - Function

```python
newmodule(name)
```

A new module with the given name.
`````

## [Wrapper types](@id julia-wrappers)

Apart from a few fundamental immutable types, all Julia values are by default converted into
Python to a [`Jl`](#juliacall.Jl) object, which wraps the original value and gives it a
Pythonic interface.

Other wrapper classes provide more specific Python semantics. For example a Julia vector can
be converted to a [`JlVector`](#juliacall.JlVector) which satisfies the Python sequence
interface and behaves very similar to a list.

- `JlBase`
  - [`Jl`](#juliacall.Jl)
  - [`JlCollection`](#juliacall.JlCollection)
    - [`JlArray`](#juliacall.JlArray)
      - [`JlVector`](#juliacall.JlVector)
    - [`JlDict`](#juliacall.JlDict)
    - [`JlSet`](#juliacall.JlSet)
  - [`JlIOBase`](#juliacall.JlIOBase)
    - `JlBinaryIO`
    - `JlTextIO`

`````@customdoc
juliacall.Jl - Class

Wraps any Julia object, giving it some basic Python semantics.

Supports `repr(x)`, `str(x)`, attributes (`x.attr`), calling (`x(a,b)`), iteration,
comparisons, `len(x)`, `a in x`, `dir(x)`.

Calling, indexing, attribute access, etc. will always return a `Jl`. To get the result
as an ordinary Python object, you can use the `.jl_to_py()` method.

Attribute access (`x.attr`) can be used to access Julia properties except those starting
and ending with `__` (since these are Python special methods) or starting with `jl_` or
`_jl_` (which are reserved by `juliacall` for Julia-specific methods).

###### Members
- `jl_callback(*args, **kwargs)`: Calls the Julia object with the given arguments.
  Unlike ordinary calling syntax, the arguments are passed as `Py` objects instead of
  being converted.
- `jl_display()`: Display the object using Julia's display mechanism.
- `jl_eval(expr)`: If the object is a Julia `Module`, evaluates the given expression.
- `jl_help()`: Display help for the object.
- `jl_to_py()`: Convert to a Python object using the [usual conversion rules](@ref jl2py).
`````

`````@customdoc
juliacall.JlCollection - Class

Wraps any Julia collection. It is a subclass of `collections.abc.Collection`.

Julia collections are arrays, sets, dicts, tuples, named tuples, refs, and in general
anything which is a collection of values in the sense that it supports functions like
`iterate`, `in`, `length`, `hash`, `==`, `isempty`, `copy`, `empty!`.

It supports `in`, `iter`, `len`, `hash`, `bool`, `==`.

###### Members
- `clear()`: Empty the collection in-place.
- `copy()`: A copy of the collection.
`````

`````@customdoc
juliacall.JlArray - Class

This wraps any Julia `AbstractArray` value. It is a subclass of
`juliacall.JlCollection`.

It supports zero-up indexing, and can be indexed with integers or slices. Slicing returns a
view of the original array.

If the array is strided and its eltype is supported (i.e. `Bool`, `IntXX`, `UIntXX`,
`FloatXX`, `Complex{FloatXX}`, `Ptr{Cvoid}` or `Tuple` or `NamedTuple` of these) then it
supports the buffer protocol and the numpy array interface. This means that
`numpy.asarray(this)` will yield a view of the original array, so mutations are visible on
the original.

Otherwise, the numpy `__array__` method is supported, and this returns an array of Python
objects converted from the contents of the array. In this case, `numpy.asarray(this)` is a
copy of the original array.

###### Members
- `ndim`: The number of dimensions.
- `shape`: Tuple of lengths in each dimension.
- `reshape(shape)`: A reshaped view of the array.
- `to_numpy(dtype=None, copy=True, order="K")`: Convert to a numpy array.
`````

`````@customdoc
juliacall.JlVector - Class

This wraps any Julia `AbstractVector` value. It is a subclass of `juliacall.JlArray` and
`collections.abc.MutableSequence` and behaves similar to a Python `list`.

###### Members
- `resize(size)`: Change the length of the vector.
- `sort(reverse=False, key=None)`: Sort the vector in-place.
- `reverse()`: Reverse the vector.
- `insert(index, value)`: Insert the value at the given index.
- `append(value)`: Append the value to the end of the vector.
- `extend(values)`: Append the values to the end of the vector.
- `pop(index=-1)`: Remove and return the item at the given index.
- `remove(value)`: Remove the first item equal to the value.
- `index(value)`: The index of the first item equal to the value.
- `count(value)`: The number of items equal to the value.
`````

`````@customdoc
juliacall.JlDict - Class
This wraps any Julia `AbstractDict` value. It is a subclass of `collections.abc.Mapping` and
behaves similar to a Python `dict`.
`````

`````@customdoc
juliacall.JlSet - Class
This wraps any Julia `AbstractSet` value. It is a subclass of `collections.abc.Set` and
behaves similar to a Python `set`.
`````

`````@customdoc
juliacall.JlIOBase - Class

This wraps any Julia `IO` value. It is a subclass of `io.IOBase` and behaves like Python
files.

There are also subtypes `JlBinaryIO` and `JlTextIO`, which are subclasses of
`io.BufferedIOBase` (buffered bytes) and `io.TextIOBase` (text).
`````
