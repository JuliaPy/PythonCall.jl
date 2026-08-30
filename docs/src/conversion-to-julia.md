# [Conversion to Julia](@id py2jl)

## [Conversion Rules](@id py2jl-conversion)

The following table specifies the conversion rules used whenever converting a Python object to a Julia object. If the initial Python type matches the "From" column, the desired type `T` intersects with the "To" column, and `T` is within the rule's scope, then that conversion is attempted. For a union target, it is sufficient for one union component to be within the scope. Matching rules are tried in reverse insertion order, so the most recently added rule is tried first.

From Julia, one can convert Python objects to a desired type using `pyconvert(T, x)` for example.

From Python, the arguments to a Julia function will be converted according to these rules with `T=Any`.

Rules with scope `Any` are canonical: they may be used by `pyconvert(Any, x)`, including
when JuliaCall passes an ordinary Python argument to Julia. Other rules are deliberately
narrower and are only considered when the caller requests a type within their scope.
PythonCall's internal wrapped-Julia and no-copy array rules are always tried before
ordinary rules.

| From                                                                                                         | To                                                          |
| :----------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------- |
| **Always-first internal conversions (scope `Any`).**                                                         |                                                             |
| `juliacall.Jl`                                                                                               | `Any`                                                       |
| Objects satisfying the buffer or array interface (inc. `bytes`, `bytearray`, `array.array`, `numpy.ndarray`) | `PyArray`                                                   |
| **Canonical conversions (scope `Any`).**                                                                     |                                                             |
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
| `numpy.datetime64`                                                                                           | `NumpyDates.DateTime64`                                     |
| `numpy.timedelta64`                                                                                          | `NumpyDates.TimeDelta64`                                    |
| **Explicitly requested conversions (scope is normally the "To" type).**                                     |                                                             |
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
| `numpy.bool_`/`numpy.intXX`/`numpy.uintXX`/`numpy.floatXX`                                                   | `Bool`, `Integer`, `Rational`, `Real`, `Number`             |
| `numpy.datetime64`                                                                                           | `NumpyDates.InlineDateTime64`, `Dates.DateTime`             |
| `numpy.timedelta64`                                                                                          | `NumpyDates.InlineTimeDelta64`, `Dates.Period`              |
| Objects satisfying the buffer or array interface                                                             | `Array`, `AbstractArray` (scope `AbstractArray`)             |
| **Fallback (scope `Any`).**                                                                                  |                                                             |
| Anything                                                                                                     | `Py`                                                        |

See [here](@ref python-wrappers) for an explanation of the `Py*` wrapper types (`PyList`, `PyIO`, etc).

## [Custom rules](@id py2jl-conversion-custom)

To add a custom conversion rule, you must define a function to do the conversion and call
`pyconvert_add_rule` to register it.

The arguments are the Python type name, the Julia target type, the Julia scope, and the
conversion function. The target type must be a subtype of the scope. You may only define
a rule when you own either the Python type or the Julia scope. In particular, use scope
`Any` only for Python types that you own. For a rule converting a foreign Python type to
your own `MyType`, use `MyType` as both target and scope:

```julia
function pyconvert_rule_mytype(::Type{T}, x::Py) where {T}
    # Construct a T, or return pyconvert_unconverted() when conversion is not possible.
    return pyconvert_return(T(x.some_attribute))
end

function __init__()
    PythonCall.pyconvert_add_rule(
        "some_package:SomePythonType",
        MyType,
        MyType,
        pyconvert_rule_mytype,
    )
end
```

This rule is considered for `pyconvert(MyType, x)`, but not for `pyconvert(Any, x)`.
Consequently, loading a package that defines such a rule does not change JuliaCall's
default conversion of unrelated Python values. When multiple matching ordinary rules
exist, the rule added most recently is tried first.

You must not do this while precompiling, so these calls will normally be in the `__init__`
function of your module.

```@docs
PythonCall.pyconvert_add_rule
```
