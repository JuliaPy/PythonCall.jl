# Conversion Rules

## Julia to Python

When a Julia object is converted to a Python one (e.g. by calling `PyObject`, by interpolating it into a `@py` command, or passing it as an argument to a Python function) the following rules are used by default.

The user can always explicitly choose a different conversion (e.g. by calling `pylist` or `pydict`).

| From                                                                | To                                              |
| :------------------------------------------------------------------ | :---------------------------------------------- |
| Any Python object type (`PyObject`, `PyList`, etc.)                 | itself                                          |
| `Nothing`, `Missing`                                                | `None`                                          |
| `Bool`                                                              | `bool`                                          |
| Standard integer (`IntXX`, `UIntXX`, `BigInt`)                      | `int`                                           |
| Standard rational (`Rational{T}`, `T` a standard integer)           | `fractions.Fraction`                            |
| Standard float (`FloatXX`)                                          | `float`                                         |
| Standard complex (`Complex{T}`, `T` a standard float)               | `complex`                                       |
| Standard string/char (`String` and `SubString{String}`, `Char`)     | `str`                                           |
| `Tuple`                                                             | `tuple`                                         |
| Standard integer range (`AbstractRange{T}`, `T` a standard integer) | `range`                                         |
| `Date`, `Time`, `DateTime` (from `Dates`)                           | `date`, `time`, `datetime` (from `datetime`)    |
| `Second`, `Millisecond`, `Microsecond`, `Nanosecond` (from `Dates`) | `timedelta` (from `datetime`)                   |
| `Number`                                                            | `julia.NumberValue`, `julia.ComplexValue`, etc. |
| `AbstractArray`                                                     | `julia.ArrayValue`, `julia.VectorValue`         |
| `AbstractDict`                                                      | `julia.DictValue`                               |
| `AbstractSet`                                                       | `julia.SetValue`                                |
| `IO`                                                                | `julia.BufferedIOValue`                         |
| `Module`                                                            | `julia.ModuleValue`                             |
| `Type`                                                              | `julia.TypeValue`                               |
| Anything else                                                       | `julia.AnyValue`                                |

The `julia.*Value` types are all subtypes of `julia.AnyValue`. They wrap a Julia value, providing access to Julia semantics: it can be called, indexed, and so on. Subtypes add additional Pythonic semantics. Read more [here](../juliapy/#Wrapper-types).

This conversion policy is defined/implemented by `Python.C.PyObject_From`.

## Python to Julia

From Julia, one can convert Python objects to a desired type using `pyconvert(T, x)` for example, or ```@pyv `...`::T```.

From Python, when a value is passed to Julia, it is typically converted to a corresponding Julia value using `pyconvert(Any, x)`.

The following table specifies the conversion rules in place. If the initial Python type matches the "from" column and the desired type `T` intersects with the "To" column, then that conversion is attempted. Conversions are tried in priority order, then in specificity order.

| From                                                                        | To         |
| :-------------------------------------------------------------------------- | :--------- |
| **Top priority (wrapped values).**                                          |            |
| `julia.AnyValue`                                                            | `Any`      |
| **Very high priority (arrays).**                                            |            |
| Objects satisfying the buffer or array interface                            | `PyArray`  |
| **High priority (canonical conversions).**                                  |            |
| ... todo ...                                                                |            |
| **Standard priority (other reasonable conversions).**                       |            |
| ... todo ...                                                                |            |
| **Low priority (fallback to `PyObject`).**                                  |            |
| Anything                                                                    | `PyObject` |
| **Bottom priority (must be explicitly specified by excluding `PyObject`).** |            |
| Objects satisfying the buffer interface                                     | `PyBuffer` |
| Anything                                                                    | `PyRef`    |

Note that as with conversion from Julia to Python, by default only immutable objects are converted to a Julia equivalent, everything else is wrapped.
