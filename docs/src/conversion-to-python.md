# [Conversion to Python](@id jl2py)

## Conversion Rules

From Julia, one converts Julia objects to Python explicitly using `Py(x)` or implicitly by passing the value to one of the many other functions, such as `pygetattr(x, "append")`.

From Python, calling a Julia function or indexing a Julia object, etc., will convert the result to some Python object.

The following table specifies the conversion rules used whenever converting a Julia object to a Python object.

The user can always explicitly choose a different conversion (e.g. by calling `pylist` or `pydict`).

| From                                                                | To                                                      |
| :------------------------------------------------------------------ | :------------------------------------------------------ |
| Any Python object type (`Py`, `PyList`, etc.)                       | itself                                                  |
| `Nothing`, `Missing`                                                | `None`                                                  |
| `Bool`                                                              | `bool`                                                  |
| Standard integer (`IntXX`, `UIntXX`, `BigInt`)                      | `int`                                                   |
| Standard rational (`Rational{T}`, `T` a standard integer)           | `fractions.Fraction`                                    |
| Standard float (`FloatXX`)                                          | `float`                                                 |
| Standard complex (`Complex{T}`, `T` a standard float)               | `complex`                                               |
| Standard string/char (`String` and `SubString{String}`, `Char`)     | `str`                                                   |
| `Tuple`                                                             | `tuple`                                                 |
| Standard integer range (`AbstractRange{T}`, `T` a standard integer) | `range`                                                 |
| `Date`, `Time`, `DateTime` (from `Dates`)                           | `date`, `time`, `datetime` (from `datetime`)            |
| `Second`, `Millisecond`, `Microsecond`, `Nanosecond` (from `Dates`) | `timedelta` (from `datetime`)                           |
| `Number`                                                            | `juliacall.NumberValue`, `juliacall.ComplexValue`, etc. |
| `AbstractArray`                                                     | `juliacall.ArrayValue`, `juliacall.VectorValue`         |
| `AbstractDict`                                                      | `juliacall.DictValue`                                   |
| `AbstractSet`                                                       | `juliacall.SetValue`                                    |
| `IO`                                                                | `juliacall.BufferedIOValue`                             |
| `Module`                                                            | `juliacall.ModuleValue`                                 |
| `Type`                                                              | `juliacall.TypeValue`                                   |
| Anything else                                                       | `juliacall.AnyValue`                                    |

See below for an explanation of the `juliacall.*Value` types.

## [Wrapper types](@id julia-wrappers)

Apart from a few fundamental immutable types, all Julia values are by default converted into Python to some [`AnyValue`](#juliacall.AnyValue) object, which wraps the original value, but giving it a Pythonic interface.

Some types are converted to a subclass of [`AnyValue`](#juliacall.AnyValue) which provides additional Python semantics --- e.g. Julia vectors are interpreted as Python sequences.

There is also a [`RawValue`](#juliacall.RawValue) object, which gives a stricter "Julia-only" interface, documented below. These types all inherit from `ValueBase`:

- `ValueBase`
  - [`RawValue`](#juliacall.RawValue)
  - [`AnyValue`](#juliacall.AnyValue)
    - [`NumberValue`](#juliacall.NumberValue)
      - `ComplexValue`
      - `RealValue`
        - `RationalValue`
        - `IntegerValue`
    - [`ArrayValue`](#juliacall.ArrayValue)
      - `VectorValue`
    - [`DictValue`](#juliacall.DictValue)
    - [`SetValue`](#juliacall.SetValue)
    - [`IOValue`](#juliacall.IOValue)
      - `BinaryIOValue`
      - `TextIOValue`
    - [`ModuleValue`](#juliacall.ModuleValue)
    - [`TypeValue`](#juliacall.TypeValue)

`````@customdoc
juliacall.AnyValue - Class

Wraps any Julia object, giving it some basic Python semantics. Subtypes provide extra semantics.

Supports `repr(x)`, `str(x)`, attributes (`x.attr`), calling (`x(a,b)`), iteration, comparisons, `len(x)`, `a in x`, `dir(x)`.

Calling, indexing, attribute access, etc. will convert the result to a Python object according to [this table](@ref jl2py). This is typically a builtin Python type (for immutables) or a subtype of `AnyValue`.

Attribute access can be used to access Julia properties as well as normal class members. In the case of a name clash, the class member will take precedence. For convenience with Julia naming conventions, `_b` at the end of an attribute is replaced with `!` and `_bb` is replaced with `!!`.

###### Members
- `_jl_raw()`: Convert to a [`RawValue`](#juliacall.RawValue). (See also [`pyjlraw`](@ref).)
- `_jl_display()`: Display the object using Julia's display mechanism.
- `_jl_help()`: Display help for the object.
`````

`````@customdoc
juliacall.NumberValue - Class

This wraps any Julia `Number` value. It is a subclass of `numbers.Number` and behaves similar to other Python numbers.

There are also subtypes `ComplexValue`, `RealValue`, `RationalValue`, `IntegerValue` which wrap values of the corresponding Julia types, and are subclasses of the corresponding `numbers` ABC.
`````

`````@customdoc
juliacall.ArrayValue - Class

This wraps any Julia `AbstractArray` value. It is a subclass of `collections.abc.Collection`.

It supports zero-up indexing, and can be indexed with integers or slices. Slicing returns a view of the original array.

There is also the subtype `VectorValue` which wraps any `AbstractVector`. It is a subclass of `collections.abc.Sequence` and behaves similar to a Python `list`.

If the array is strided and its eltype is supported (i.e. `Bool`, `IntXX`, `UIntXX`, `FloatXX`, `Complex{FloatXX}`, `Ptr{Cvoid}` or `Tuple` or `NamedTuple` of these) then it supports the buffer protocol and the numpy array interface. This means that `numpy.asarray(this)` will yield a view of the original array, so mutations are visible on the original.

Otherwise, the numpy `__array__` method is supported, and this returns an array of Python objects converted from the contents of the array. In this case, `numpy.asarray(this)` is a copy of the original array.

###### Members
- `ndim`: The number of dimensions.
- `shape`: Tuple of lengths in each dimension.
- `copy()`: A copy of the array.
- `reshape(shape)`: A reshaped view of the array.
`````

`````@customdoc
juliacall.DictValue - Class
This wraps any Julia `AbstractDict` value. It is a subclass of `collections.abc.Mapping` and behaves similar to a Python `dict`.
`````

`````@customdoc
juliacall.SetValue - Class
This wraps any Julia `AbstractSet` value. It is a subclass of `collections.abc.Set` and behaves similar to a Python `set`.
`````

`````@customdoc
juliacall.IOValue - Class

This wraps any Julia `IO` value. It is a subclass of `io.IOBase` and behaves like Python files.

There are also subtypes `BinaryIOValue` and `TextIOValue`, which are subclasses of `io.BufferedIOBase` (buffered bytes) and `io.TextIOBase` (text).
`````

`````@customdoc
juliacall.ModuleValue - Class
This wraps any Julia `Module` value.

It is the same as [`AnyValue`](#juliacall.AnyValue) except for one additional convenience method:
- `seval([module=self], code)`: Evaluates the given code (a string) in the given module.
`````

`````@customdoc
juliacall.TypeValue - Class

This wraps any Julia `Type` value.

It is the same as [`AnyValue`](#juliacall.AnyValue) except that indexing is used to access Julia's "curly" syntax for specifying parametric types:

```python
from juliacall import Main as jl
# equivalent to Vector{Int}() in Julia
jl.Vector[jl.Int]()
```
`````

`````@customdoc
juliacall.RawValue - Class

Wraps any Julia value with a rigid interface suitable for generic programming.

Supports `repr(x)`, `str(x)`, attributes (`x.attr`), calling (`x(a,b)`), `len(x)`, `dir(x)`.

This is very similar to [`AnyValue`](#juliacall.AnyValue) except that indexing, calling, etc. will always return a `RawValue`.

Indexing with a tuple corresponds to indexing in Julia with multiple values. To index with a single tuple, it will need to be wrapped in another tuple.

###### Members
- `_jl_any()`: Convert to a [`AnyValue`](#juliacall.AnyValue) (or subclass). (See also [`pyjl`](@ref).)
`````
