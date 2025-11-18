# [Conversion to Julia](@id py2jl)

## [Conversion Rules](@id py2jl-conversion)

The following table specifies the conversion rules used whenever converting a Python object to a Julia object. If the initial Python type matches the "From" column and the desired type `T` intersects with the "To" column, then that conversion is attempted. Rules are ordered by Python type specificity (strict subclassing only) and then by creation order. A rule only applies when the requested target type is a subtype of its scope; the "Scope" column lists the type that must match.

From Julia, one can convert Python objects to a desired type using `pyconvert(T, x)` for example.

From Python, the arguments to a Julia function will be converted according to these rules with `T=Any`.

| From                                                                                                         | To                                             | Scope |
| :----------------------------------------------------------------------------------------------------------- | :--------------------------------------------- | :---- |
| **Default-scope rules (apply even when converting to `Any`).**                                              |                                                |       |
| `juliacall.Jl`                                                                                               | `Any`                                          | `Any` |
| Objects satisfying the buffer or array interface (inc. `bytes`, `bytearray`, `array.array`, `numpy.ndarray`) | `PyArray`                                      | `Any` |
| `None`                                                                                                       | `Nothing`                                      | `Any` |
| `bool`                                                                                                       | `Bool`                                         | `Any` |
| `numbers.Integral` (inc. `int`)                                                                              | `Integer` (prefers `Int`, or `BigInt` on overflow) | `Any` |
| `numbers.Rational`                                                                                           | `Rational{<:Integer}`                           | `Any` |
| `float`                                                                                                      | `Float64`                                      | `Any` |
| `complex`                                                                                                    | `Complex{Float64}`                             | `Any` |
| `range`                                                                                                      | `StepRange{<:Integer,<:Integer}`               | `Any` |
| `str`                                                                                                        | `String`                                       | `Any` |
| `bytes`                                                                                                      | `Base.CodeUnits{UInt8,String}`                  | `Any` |
| `tuple`                                                                                                      | `NamedTuple`                                   | `Any` |
| `tuple`                                                                                                      | `Tuple`                                        | `Any` |
| `datetime.datetime`                                                                                          | `DateTime`                                     | `Any` |
| `datetime.date`                                                                                              | `Date`                                         | `Any` |
| `datetime.time`                                                                                              | `Time`                                         | `Any` |
| `datetime.timedelta`                                                                                         | `Microsecond`                                  | `Any` |
| `numpy.bool_`/`numpy.intXX`/`numpy.uintXX`/`numpy.floatXX`/`numpy.complexXX`                                  | matching Julia scalar                           | `Any` |
| `numpy.datetime64`                                                                                           | `NumpyDates.DateTime64`                        | `Any` |
| `numpy.timedelta64`                                                                                          | `NumpyDates.TimeDelta64`                       | `Any` |
| `pandas._libs.missing.NAType`                                                                                | `Missing`                                      | `Any` |
| `juliacall.JlBase`                                                                                           | `Any`                                          | `Any` |
| `builtins.object`                                                                                            | `Py`                                           | `Any` |
| **Scoped conversions (only when the requested target matches the scope).**                                   |                                                |       |
| `None`                                                                                                       | `Missing`                                      | `Missing` |
| `bool`                                                                                                       | `Number`                                       | `Number` |
| `float`                                                                                                      | `Number`                                       | `Number` |
| `float` (NaN)                                                                                                | `Nothing` / `Missing`                          | `Nothing` / `Missing` |
| `complex`                                                                                                    | `Number`                                       | `Number` |
| `numbers.Integral`                                                                                           | `Number`                                       | `Number` |
| `numbers.Rational`                                                                                           | `Number`                                       | `Number` |
| `str`                                                                                                        | `Symbol`                                       | `Symbol` |
| `str`                                                                                                        | `Char`                                         | `Char` |
| `bytes`                                                                                                      | `Vector{UInt8}`                                | `Vector{UInt8}` |
| `range`                                                                                                      | `UnitRange{<:Integer}`                         | `UnitRange{<:Integer}` |
| `collections.abc.Iterable`                                                                                   | `Vector` / `Tuple` / `Set` / `NamedTuple` / `Pair` | respective targets |
| `collections.abc.Sequence`                                                                                   | `Vector` / `Tuple`                             | respective targets |
| `collections.abc.Set`                                                                                        | `Set`                                          | `Set` |
| `collections.abc.Mapping`                                                                                    | `Dict`                                         | `Dict` |
| `datetime.timedelta`                                                                                         | `Millisecond` / `Second` / `Nanosecond`        | same as target |
| `numpy.datetime64`                                                                                           | `NumpyDates.InlineDateTime64`                  | `NumpyDates.InlineDateTime64` |
| `numpy.datetime64`                                                                                           | `NumpyDates.DatesInstant`                      | `NumpyDates.DatesInstant` |
| `numpy.datetime64`                                                                                           | `Missing` / `Nothing`                          | same as target |
| `numpy.timedelta64`                                                                                          | `NumpyDates.InlineTimeDelta64`                 | `NumpyDates.InlineTimeDelta64` |
| `numpy.timedelta64`                                                                                          | `NumpyDates.DatesPeriod`                       | `NumpyDates.DatesPeriod` |
| `numpy.timedelta64`                                                                                          | `Missing` / `Nothing`                          | same as target |
| NumPy scalars (`numpy.bool_`, `numpy.intXX`, `numpy.uintXX`, `numpy.floatXX`, `numpy.complexXX`)             | `Int` / `UInt` / `Integer` / `Real` / `Complex{Float64}` / `Complex` / `Number` | scope matches target |
| ctypes simple values (`c_int`, `c_double`, `c_void_p`, `c_char_p`, etc.)                                     | matching C type; widening numeric targets (`Int` / `UInt` / `Integer` / `Real` / `Number`); pointers (`Ptr`, `Cstring`, `Cwstring`) | scope matches target |
| `pandas._libs.missing.NAType`                                                                                | `Nothing`                                      | `Nothing` |
| **Wrapper conversions (opt-in scopes).**                                                                     |                                                |       |
| `collections.abc.Iterable`                                                                                   | `PyIterable`                                   | `PyIterable` |
| `collections.abc.Sequence`                                                                                   | `PyList`                                       | `PyList` |
| `collections.abc.Set`                                                                                        | `PySet`                                        | `PySet` |
| `collections.abc.Mapping`                                                                                    | `PyDict`                                       | `PyDict` |
| `io.IOBase` / `_io._IOBase`                                                                                  | `PyIO`                                         | `PyIO` |
| `pandas.core.frame.DataFrame`                                                                                | `PyPandasDataFrame`                            | `PyPandasDataFrame` |
| `pandas.core.arrays.base.ExtensionArray`                                                                     | `PyList`                                       | `PyList` |
| Objects satisfying the buffer or array interface                                                             | `Array` / `AbstractArray`                      | same as target |
| **Explicit wrapper opt-out.**                                                                                |                                                |       |
| Anything                                                                                                     | `PyRef`                                        | `PyRef` |

See [here](@ref python-wrappers) for an explanation of the `Py*` wrapper types (`PyList`, `PyIO`, etc).

## [Custom rules](@id py2jl-conversion-custom)

To add a custom conversion rule, you must define a function to do the conversion and call
`pyconvert_add_rule` to register it.

You must not do this while precompiling, so these calls will normally be in the `__init__`
function of your module.

```@docs
PythonCall.pyconvert_add_rule
```
