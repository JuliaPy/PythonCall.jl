![Python.jl logo](https://raw.githubusercontent.com/cjdoris/Python.jl/master/logo-text.svg)
---

Call Python from Julia!

## Basic usage by example

```julia
# import a python module
pymath = pyimport("math")

# julia property access <--> python attribute access
println("pi = ", pymath.pi)

# basic julia types (numbers, strings, etc) automatically converted to their Julia equivalent
pymath.sin(3.141) + pymath.cos(3.141)
```

## Conversion from Julia to Python

Julia objects are converted to a Python equivalent automatically when required.

The function `pyobject(x)` implements this conversion.

- `nothing` becomes `None`
- Boolean, integer, floating-point, rational and complex numbers become their Python equivalent
- Integer ranges become `range`
- Strings and characters become `str`
- Tuples and pairs become `tuple`
- Dates, times and datetimes become their equivalent from `datetime`

Everything else is wrapped into a Python wrapper around the Julia value (implemented by `pyjl(x)`). This wrapper implements Python interfaces where appropriate, including subclassing the appropriate abstract base class:

- Property access, indexing, function calls, arithmetic, etc. behaves as expected
- Iterable objects implement the iterable interface
- Vectors implement the sequence interface
- Dictionaries implement the mapping interface
- Sets implement the set interface
- Arrays with suitable element type implement the buffer interface
- Numbers implement the numbers interface
- IO objects implement the binary IO interface (call `pytextio(x)` for a text IO wrapper instead)

## Conversion from Python to Julia

Calling Python functions always returns a Python object; no automatic conversion back to Julia is performed.

Use `pyconvert(T, o)` to convert the Python object `o` to a Julia object of type `T`.
