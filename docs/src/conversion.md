# Conversion Rules

This page documents the rules used to convert values between Julia and Python.

In both directions, the default behaviour is to allow conversion between immutable values. Mutable values will be "wrapped" so that mutations on the wrapper affect the original object.

## [Julia to Python](@id jl2py)

When a Julia object is converted to a Python one (e.g. by calling `PyObject`, by interpolating it into a `@py` command, or passing it as an argument to a Python function) the following rules are used by default.

The user can always explicitly choose a different conversion (e.g. by calling `pylist` or `pydict`).

| From                                                                | To                                                  |
| :------------------------------------------------------------------ | :-------------------------------------------------- |
| Any Python object type (`PyObject`, `PyList`, etc.)                 | itself                                              |
| `Nothing`, `Missing`                                                | `None`                                              |
| `Bool`                                                              | `bool`                                              |
| Standard integer (`IntXX`, `UIntXX`, `BigInt`)                      | `int`                                               |
| Standard rational (`Rational{T}`, `T` a standard integer)           | `fractions.Fraction`                                |
| Standard float (`FloatXX`)                                          | `float`                                             |
| Standard complex (`Complex{T}`, `T` a standard float)               | `complex`                                           |
| Standard string/char (`String` and `SubString{String}`, `Char`)     | `str`                                               |
| `Tuple`                                                             | `tuple`                                             |
| Standard integer range (`AbstractRange{T}`, `T` a standard integer) | `range`                                             |
| `Date`, `Time`, `DateTime` (from `Dates`)                           | `date`, `time`, `datetime` (from `datetime`)        |
| `Second`, `Millisecond`, `Microsecond`, `Nanosecond` (from `Dates`) | `timedelta` (from `datetime`)                       |
| `Number`                                                            | `juliacall.NumberValue`, `juliacall.ComplexValue`, etc. |
| `AbstractArray`                                                     | `juliacall.ArrayValue`, `juliacall.VectorValue`         |
| `AbstractDict`                                                      | `juliacall.DictValue`                                 |
| `AbstractSet`                                                       | `juliacall.SetValue`                                  |
| `IO`                                                                | `juliacall.BufferedIOValue`                           |
| `Module`                                                            | `juliacall.ModuleValue`                               |
| `Type`                                                              | `juliacall.TypeValue`                                 |
| Anything else                                                       | `juliacall.AnyValue`                                  |

The `juliacall.*Value` types are all subtypes of `juliacall.AnyValue`. They wrap a Julia value, providing access to Julia semantics: it can be called, indexed, and so on. Subtypes add additional Pythonic semantics. Read more [here](@ref julia-wrappers).

This conversion policy is defined/implemented by `PythonCall.C.PyObject_From` and `PythonCall.C.PyJuliaValue_From`. Package authors can (carefully) overload these with additional rules for custom types.

## [Python to Julia](@id py2jl)

From Julia, one can convert Python objects to a desired type using `pyconvert(T, x)` for example, or ```@pyv `...`::T```.

From Python, when a value is passed to Julia, it is typically converted to a corresponding Julia value using `pyconvert(Any, x)`.

Quite general conversions are allowed, and the target type `T` can be as specific as you like. For example
```
@pyv `[1, None, 3]`::Tuple{Vararg{Union{AbstractFloat,Missing}}}
```
evaluates to `(1.0, missing, 2.0)`.

The following table specifies the conversion rules in place. If the initial Python type matches the "From" column and the desired type `T` intersects with the "To" column, then that conversion is attempted. Conversions are tried in priority order, then in specificity order.

| From                                                                                                         | To                                                          |
| :----------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------- |
| **Top priority (wrapped values).**                                                                           |                                                             |
| `juliacall.AnyValue`                                                                                           | `Any`                                                       |
| **Very high priority (arrays).**                                                                             |                                                             |
| Objects satisfying the buffer or array interface (inc. `bytes`, `bytearray`, `array.array`, `numpy.ndarray`) | `PyArray`                                                   |
| **High priority (canonical conversions).**                                                                   |                                                             |
| `None`                                                                                                       | `Nothing`                                                   |
| `bool`                                                                                                       | `Bool`                                                      |
| `numbers.Integral` (inc. `int`)                                                                              | `Integer` (prefers `Int`, or `BigInt` on overflow)          |
| `float`                                                                                                      | `Float64`                                                   |
| `complex`                                                                                                    | `Complex{Float64}`                                          |
| `range`                                                                                                      | `StepRange`                                                 |
| `str`                                                                                                        | `String`                                                    |
| `tuple`                                                                                                      | `Tuple`                                                     |
| `collections.abc.Mapping` (inc. `dict`)                                                                      | `PyDict`                                                    |
| `collections.abc.Sequence` (inc. `list`)                                                                     | `PyList`                                                    |
| `collections.abc.Set` (inc. `set`, `frozenset`)                                                              | `PySet`                                                     |
| `io.IOBase` (includes open files)                                                                            | `PyIO`                                                      |
| `datetime.date`/`datetime.time`/`datetime.datetime`                                                          | `Date`/`Time`/`DateTime`                                    |
| `datetime.timedelta`                                                                                         | `Microsecond` (or `Millisecond` or `Second` on overflow)    |
| `numpy.intXX`/`numpy.uintXX`/`numpy.floatXX`                                                                 | `IntXX`/`UIntXX`/`FloatXX`                                  |
| **Standard priority (other reasonable conversions).**                                                        |                                                             |
| `None`                                                                                                       | `Missing`                                                   |
| `bytes`                                                                                                      | `Vector{UInt8}`, `Vector{Int8}`, `String`                   |
| `str`                                                                                                        | `String`, `Symbol`, `Char`, `Vector{UInt8}`, `Vector{Int8}` |
| `range`                                                                                                      | `UnitRange`                                                 |
| `collections.abc.Mapping`                                                                                    | `Dict`                                                      |
| `collections.abc.Iterable`                                                                                   | `Vector`, `Set`, `Tuple`, `NamedTuple`, `Pair`              |
| `datetime.timedelta`                                                                                         | `Dates.CompoundPeriod`                                      |
| `numbers.Integral`                                                                                           | `Integer`, `Rational`, `Real`, `Number`                     |
| `numbers.Real`                                                                                               | `AbstractFloat`, `Number`                                   |
| `numbers.Complex`                                                                                            | `Complex`, `Number`                                         |
| `ctypes.c_int` and other integers                                                                            | `Integer`, `Rational`, `Real`, `Number`                     |
| `ctypes.c_float`/`ctypes.c_double`                                                                           | `Cfloat`/`Cdouble`, `AbstractFloat`, `Real`, `Number`       |
| `ctypes.c_voidp`                                                                                             | `Ptr{Cvoid}`, `Ptr`                                         |
| `ctypes.c_char_p`                                                                                            | `Cstring`, `Ptr{Cchar}`, `Ptr`                              |
| `ctypes.c_wchar_p`                                                                                           | `Cwstring`, `Ptr{Cwchar}`, `Ptr`                            |
| `numpy.intXX`/`numpy.uintXX`/`numpy.floatXX`                                                                 | `Integer`, `Rational`, `Real`, `Number`                     |
| **Low priority (fallback to `PyObject`).**                                                                   |                                                             |
| Anything                                                                                                     | `PyObject`                                                  |
| **Bottom priority (must be explicitly specified by excluding `PyObject`).**                                  |                                                             |
| Objects satisfying the buffer interface                                                                      | `PyBuffer`                                                  |
| Anything                                                                                                     | `PyRef`                                                     |

Package authors can (carefully) add extra rules by calling `PythonCall.C.PyObject_TryConvert_AddRule` in `__init__`.
