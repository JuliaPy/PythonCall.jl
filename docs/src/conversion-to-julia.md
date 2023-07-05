# [Conversion to Julia](@id py2jl)

## [Conversion Rules](@id py2jl-conversion)

The following table specifies the conversion rules used whenever converting a Python object to a Julia object. If the initial Python type matches the "From" column and the desired type `T` intersects with the "To" column, then that conversion is attempted. Conversions are tried in priority order, then in specificity order.

From Julia, one can convert Python objects to a desired type using `pyconvert(T, x)` for example.

From Python, the arguments to a Julia function will be converted according to these rules with `T=Any`.

| From                                                                                                         | To                                                          |
| :----------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------- |
| **Top priority (wrapped values).**                                                                           |                                                             |
| `juliacall.AnyValue`                                                                                         | `Any`                                                       |
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
| `BaseException`                                                                                              | `PyException`                                               |
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
| `numbers.Real`                                                                                               | `AbstractFloat`, `Number`, `Missing`/`Nothing` (if NaN)     |
| `numbers.Complex`                                                                                            | `Complex`, `Number`                                         |
| `ctypes.c_int` and other integers                                                                            | `Integer`, `Rational`, `Real`, `Number`                     |
| `ctypes.c_float`/`ctypes.c_double`                                                                           | `Cfloat`/`Cdouble`, `AbstractFloat`, `Real`, `Number`       |
| `ctypes.c_voidp`                                                                                             | `Ptr{Cvoid}`, `Ptr`                                         |
| `ctypes.c_char_p`                                                                                            | `Cstring`, `Ptr{Cchar}`, `Ptr`                              |
| `ctypes.c_wchar_p`                                                                                           | `Cwstring`, `Ptr{Cwchar}`, `Ptr`                            |
| `numpy.intXX`/`numpy.uintXX`/`numpy.floatXX`                                                                 | `Integer`, `Rational`, `Real`, `Number`                     |
| Objects satisfying the buffer or array interface                                                             | `Array`, `AbstractArray`                                    |
| **Low priority (fallback to `Py`).**                                                                         |                                                             |
| Anything                                                                                                     | `Py`                                                        |
| **Bottom priority (must be explicitly specified by excluding `Py`).**                                        |                                                             |
| Objects satisfying the buffer interface                                                                      | `PyBuffer`                                                  |
| Anything                                                                                                     | `PyRef`                                                     |

See [here](@ref python-wrappers) for an explanation of the `Py*` wrapper types (`PyList`, `PyIO`, etc).

## [Custom rules](@id py2jl-conversion-custom)

To add a custom conversion rule, you must define a function to do the conversion and call
`pyconvert_add_rule` to register it.

You must not do this while precompiling, so these calls will normally be in the `__init__`
function of your module.

```@docs
PythonCall.pyconvert_add_rule
```
