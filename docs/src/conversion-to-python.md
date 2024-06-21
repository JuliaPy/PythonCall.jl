# [Conversion to Python](@id jl2py)

## [Conversion Rules](@id jl2py-conversion)

The following table specifies the conversion rules used whenever converting a Julia object to a Python object.

From Julia, this occurs explicitly with `Py(x)` or implicitly when passing Julia objects as the argument to a Python function.
To avoid this automatic conversion, the user can convert objects explicitly, such as by calling `pylist` or `pydict`.

From Python, this occurs when converting the return value of a Julia function.

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
