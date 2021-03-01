# The Python module `juliacall`

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

When a Python value is passed to Julia, then typically it will be converted according to [this table](@ref py2jl) with `T=Any`. Sometimes a more specific type will be used, such as when assigning to an array whose element type is known.

When a Julia value is returned to Python, it will normally be converted according to [this table](@ref jl2py).

## [Wrapper types](@id julia-wrappers)

Apart from a few fundamental immutable types (see [here](@ref jl2py)), all Julia values are by default converted into Python to some [`AnyValue`](#juliacall.AnyValue) object, which wraps the original value. Some types are converted to a subclass of [`AnyValue`](#juliacall.AnyValue) which provides additional Python semantics --- e.g. Julia vectors are interpreted as Python sequences.

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
      - `RawIOValue`
      - `BufferedIOValue`
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

There are also subtypes `RawIOValue`, `BufferedIOValue` and `TextIOValue`, which are subclasses of `io.RawIOBase` (unbuffered bytes), `io.BufferedIOBase` (buffered bytes) and `io.TextIOBase` (text).

###### Members
- `torawio()`: Convert to a `RawIOValue`, an un-buffered bytes file-like object. (See also [`pyrawio`](@ref).)
- `tobufferedio()`: Convert to a `BufferedIOValue`, an buffered bytes file-like object. Julia `IO` objects are converted to this by default. (See also [`pybufferedio`](@ref).)
- `totextio()`: Convert to a `TextIOValue`, a text file-like object. (See also [`pytextio`](@ref).)
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
