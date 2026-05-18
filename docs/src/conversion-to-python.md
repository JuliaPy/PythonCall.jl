# [Conversion to Python](@id jl2py)

## [Conversion Rules](@id jl2py-conversion)

The following table specifies the conversion rules used whenever converting a Julia object to a Python object.

From Julia, this occurs explicitly with `Py(x)` or implicitly when passing Julia objects as the argument to a Python function.
To avoid this automatic conversion, the user can convert objects explicitly, such as by calling `pylist` or `pydict`.

From Python, this occurs when converting the return value of a Julia function.

| From                                                                | To                                                      |
| :------------------------------------------------------------------ | :------------------------------------------------------ |
| Any Python object type (`Py`, `PyList`, etc.)                       | itself                                                  |
| `Nothing`                                                           | `None`                                                  |
| `Bool`                                                              | `bool`                                                  |
| `Integer`                                                           | `int`                                                   |
| `Rational{<:Integer}`                                               | `fractions.Fraction`                                    |
| `Float64`, `Float32`, `Float16`                                     | `float`                                                 |
| `Complex{Float64}`, `Complex{Float32}`, `Complex{Float16}`          | `complex`                                               |
| `AbstractString`, `AbstractChar`                                    | `str`                                                   |
| `Base.CodeUnits{UInt8}` (e.g. `b"example"`)                         | `bytes`                                                 |
| `Tuple`, `Pair`                                                     | `tuple`                                                 |
| `AbstractRange{<:Integer}`                                          | `range`                                                 |
| `Dates.Date`                                                        | `datetime.date`                                         |
| `Dates.Time`                                                        | `datetime.time`                                         |
| `Dates.DateTime`                                                    | `datetime.datetime`                                     |
| `Dates.Second`, `Dates.Millisecond`, `Dates.Microsecond`, `Dates.Nanosecond` | `datetime.timedelta`                           |
| `AbstractArray`                                                     | `juliacall.JlArray`, `juliacall.JlVector`               |
| `AbstractDict`                                                      | `juliacall.JlDict`                                      |
| `AbstractSet`                                                       | `juliacall.JlSet`                                       |
| `IO`                                                                | `juliacall.JlBinaryIO`                                  |
| Anything else                                                       | `juliacall.Jl`                                          |

See [here](@ref julia-wrappers) for an explanation of the `juliacall.Jl*` wrapper types.

## [Custom rules](@id jl2py-conversion-custom)

You may define a new conversion rule for your new type `T` by overloading `Py(::T)`.

If `T` is a wrapper type (such as `PyList`) where `Py(x)` simply returns the stored Python
object, then also define `ispy(::T) = true`.

```@docs
PythonCall.ispy
```
