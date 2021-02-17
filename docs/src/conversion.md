# Conversion Rules

## Julia to Python

When a Julia object is converted to a Python one (e.g. by calling `PyObject`, by interpolating it into a `@py` command, or passing it as an argument to a Python function) the following rules are used by default.

The user can always explicitly choose a different conversion (e.g. by calling `pylist` or `pydict`).

| From | To |
| :--- | :- |
| Any Python object type | itself |
| `nothing`, `missing` | `None` |
| `Bool` | `bool` |
| Standard integer (`IntXX`, `UIntXX`, `BigInt`) | `int` |
| Standard rational (`Rational{T}` where `T` is a standard integer) | `fractions.Fraction` |
| Standard float (`FloatXX`) | `float` |
| Standard complex (`Complex{T}` where `T` is a standard float) | `complex` |
| Standard string/char (`String` and `SubString{String}`, `Char`) | `str` |
| `Tuple` | `tuple` |
| Standard integer range (`AbstractRange{T}` where `T` is a standard integer) | `range` |
| `Date`, `Time`, `DateTime` (from `Dates`) | `date`, `time`, `datetime` (from `datetime`) |
| `Second`, `Millisecond`, `Microsecond`, `Nanosecond` (from `Dates`) | `timedelta` (from `datetime`) |
| Anything else | `julia.AnyValue` (read below) |

This conversion policy is defined/implemented by `Python.C.PyObject_From`.

The type `julia.AnyValue` holds a reference to a Julia object and provides access to Julia semantics about the object: it can be called, indexed, and so on.

There are also subtypes, which have additional Pythonic semantics.

| From | To |
| :--- | :- |
| `Any` | `julia.AnyValue` |
| `Integer`, `Rational`, `Real`, `Complex`, `Number` | `julia.IntegerValue`, `julia.RationalValue`, `julia.RealValue`, `julia.ComplexValue`, `julia.NumberValue`. These fully implement the corresponding numeric interfaces from the `numbers` package. |
| `AbstractArray`, `AbstractVector` | `julia.ArrayValue`, `julia.VectorValue`. These implement the `Collection` and `Sequence` interfaces from `collections.abc`. If the underlying array is strided and the element type is sufficiently compatible, the data is exposed through the buffer interface and numpy array interface. Always provides `__array__`, falling back to returning an array of objects, so conversion to numpy is always possible. |
| `AbstractDict`, `AbstractSet` | `julia.DictValue`, `julia.SetValue`. These implement the `Mapping` and `Set` interfaces from `collections.abc`, and behave like `dict` and `set`. |
| `IO` | `julia.BufferedIOValue`. Implements the `BufferedIOBase` interface from `io`. You can use `pytextio` to create a `julia.TextIOValue` instead, which implements `TextIOBase`. |
| `Module` | `julia.ModuleValue`. This has one special method `seval` which takes a string of Julia code and evaluates it. |
| `Type` | `julia.TypeValue`. This over-rides indexing behaviour so that indexing can be used to access the "curly" syntax for specifying parametric types. |

## Python to Julia

From Julia, one can convert Python objects to a desired type using `pyconvert(T, x)` for example, or ```@pyv `...`::T```.

From Python, when a value is passed to Julia, it is typically converted to a corresponding Julia value using `pyconvert(Any, x)`.

The following table specifies the conversion rules in place. If the initial Python type matches the "from" column and the desired type `T` intersects with the "To" column, then that conversion is attempted. Conversions are tried in priority order, then in specificity order.

| From | To | How |
| :--- | :- | :-- |
| **Top priority (wrapped values).** |||
| `julia.AnyValue` | `Any` | Extract the Julia value and try to `convert`. |
| **Very high priority (arrays).** |||
| Objects satisfying the buffer or array interface | `PyArray` ||
| **High priority (canonical conversions).** |||
| ... todo ... |||
| **Standard priority (other reasonable conversions).** |||
| ... todo ... |||
| **Low priority (fallback to `PyObject`).**|||
| Anything | `PyObject` ||
| **Bottom priority (must be explicitly specified by excluding `PyObject`).** |||
| Objects satisfying the buffer interface | `PyBuffer` ||
| Anything | `PyRef` ||

Note that as with conversion from Julia to Python, by default only immutable objects are converted to a Julia equivalent, everything else is wrapped.
