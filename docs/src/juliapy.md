# Julia from Python

## Wrapper types

Apart from a few fundamental immutable types (see [here](../conversion/#Julia-to-Python)), all Julia values are by default converted into Python to some `julia.AnyValue` object, which wraps the original value. Some types are converted to a subclass of `julia.AnyValue` which provides additional Python semantics --- e.g. Julia vectors are interpreted as Python sequences.

There is also a `julia.RawValue` object, which gives a stricter "Julia-only" interface, documented below. These types all inherit from `julia.ValueBase`.

### `julia.AnyValue`

#### Members
- `__jl_raw()`: Convert to a `julia.RawValue`. (See also [`pyjlraw`](@ref).)

```@raw html
<article class="docstring">
    <header>
        <a class="docstring-binding" id="julia.NumberValue" href="#julia.NumberValue">julia.NumberValue</a>
        —
        <span class="docstring-category">Python Class</span>
    </header>
    <section>
        <p>This wraps any Julia <code>Number</code> value. It is a subclass of <code>numbers.Number</code> and behaves similar to other Python numbers.</p>
        <p>There are also subtypes <code>julia.ComplexValue</code>, <code>julia.RealValue</code>, <code>julia.RationalValue</code>, <code>julia.IntegerValue</code> which wrap values of the corresponding Julia types, and are subclasses of the corresponding <code>numbers</code> ABC.</p>
    </section>
</article>
```

### `julia.NumberValue` old

This wraps any Julia `Number` value. It is a subclass of `numbers.Number` and behaves similar to other Python numbers.

There are also subtypes `julia.ComplexValue`, `julia.RealValue`, `julia.RationalValue` and `julia.IntegerValue`, which wrap values of the corresponding Julia types, and are subclasses of the corresponding `numbers` ABC.

### `julia.ArrayValue`

This wraps any Julia `AbstractArray` value. It is a subclass of `collections.abc.Collection`.

It supports zero-up indexing, and can be indexed with integers or slices. Slicing returns a view of the original array.

There is also the subtype `julia.VectorValue` which wraps any `AbstractVector`. It is a subclass of `collections.abc.Sequence` and behaves similar to a Python `list`.

If the array is strided and its eltype is supported (i.e. it is a `Bool`, `IntXX`, `UIntXX`, `FloatXX`, `Complex{FloatXX}`, `Ptr{Cvoid}` or tuple or named tuple of these) then it supports the buffer protocol and the numpy array interface. This means that `numpy.asarray(this)` will yield a view of the original array, so mutations are visible on the original.

Otherwise, the numpy `__array__` method is supported, and this returns an array of Python objects converted from the contents of the array. In this case, `numpy.asarray(this)` is a copy of the original array.

#### Members
- `ndim`: the number of dimensions.
- `shape`: tuple of lengths in each dimension.
- `copy()`: return a copy of the array.
- `reshape(shape)`: a reshaped view of the array.

### `julia.DictValue`

This wraps any Julia `AbstractDict` value. It is a subclass of `collections.abc.Mapping` and behaves similar to a Python `dict`.

### `julia.SetValue`

This wraps any Julia `AbstractSet` value. It is a subclass of `collections.abc.Set` and behaves similar to a Python `set`.

### `julia.IOValue`

This wraps any Julia `IO` value. It is a subclass of `io.IOBase`.

There are also subtypes `julia.RawIOValue`, `julia.BufferedIOValue` and `julia.TextIOValue`, which are subclasses of `io.RawIOBase` (unbuffered bytes), `io.BufferedIOBase` (buffered bytes) and `io.TextIOBase` (text).

#### Members
- `torawio()`: Convert to a `julia.RawIOValue`, an un-buffered file-like object. (See also [`pyrawio`](@ref).)
- `tobufferedio()`: Convert to a `julia.BufferedIOValue`, a byte-based file-like object. Julia `IO` objects are converted to this by default. (See also [`pybufferedio`](@ref).)
- `totextio()`: Convert to a `julia.TextIOValue`, a text-based file-like object. (See also [`pytextio`](@ref).)

### `julia.ModuleValue`

This wraps any Julia `Module` value.

It is the same as `julia.AnyValue` except for one additional convenience method:

- `seval([module=self], code)`: Evaluates the given code (a string) in the given module.

### `julia.TypeValue`

This wraps any Julia `Type` value.

It is the same as `julia.AnyValue` except that indexing is used to access Julia's "curly" syntax for specifying parametric types:

```python
from julia import Main as jl
jl.Vector[jl.Int]() # equivalent to Vector{Int}() in Julia
```

### `julia.RawValue`

This can wrap any Julia value.

#### Members
- `__jl_any()`: Convert to a `julia.AnyValue` (or subclass). (See also [`pyjl`](@ref).)
