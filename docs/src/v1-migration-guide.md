# [v0.9 to v1 Migration Guide](@id v1-migration-guide)

## Core functionality

Comparisons (`==`, `<`, etc.) between Python objects `Py`, or between `Py` and `Number`,
used to return `Py` but now return `Bool`. The old behaviour was a pun but broke the
Base API behaviour of these functions. These comparisons will now raise an error if the
underlying Python operation does not return `bool`.

* Instead of `pytruth(Py(3) < Py(5))` use `Py(3) < Py(5)`.
* Instead of `Py(3) < Py(5)` use `Py(Py(3) < Py(5))`.
* Instead of `np.array([1,2,3]) < Py(3)` use `pylt(np.array([1,2,3]), Py(3))`. This is
  because comparisons on numpy arrays return arrays of `bool` rather than a single
  `bool`.
* Instead of `pylt(Bool, Py(3), Py(5))` you can use `Py(3) < Py(5)`.

## `PythonCall.GC`

This submodule has been changed to closer mimic the `Base.GC` API.

* Instead of `PythonCall.GC.enable()` use `PythonCall.GC.enable(true)`.
* Instead of `PythonCall.GC.disable()` use `PythonCall.GC.enable(false)`.

## Python wrappers (`PyArray`, etc.)

`PyArray` has been reparametrised from `PyArray{T,N,L,M,R}` to `PyArray{T,N,F}` where
`F` is a `Tuple` of `Symbol` flags replacing `L` (now `:linear`) and `M`
(now `:mutable`). The `R` parameter (the underlying raw type) is removed and now implied
by `T`.

* Instead of `PyArray{Int,2,true,true,Int}` use `PyArray{Int,2,(:mutable,:linear)}`.
* Instead of `PyArray{Bool,1,false,false,Bool}` use `PyArray{Bool,1,()}`.
* Instead of `PyArray{Py,2,false,false,PythonCall.Wrap.UnsafePyObject}` use `PyArray{Py,2,()}`.

Because the `R` parameter is removed, if the underlying array is of Python objects, the
`PyArray` must have eltype `Py`. Previously you could construct a `PyArray{String}` from
such a thing and the elements would be automatically `pyconvert(String, element)`-ed for
you.

* Instead of `PyArray{String}(x)` use `pyconvert.(String, PyArray{Py}(x))` if you are
  OK with taking a copy. Or use `mappedarray(x->pyconvert(String, x), PyArray{Py}(x))`
  from [MappedArrays.jl](https://github.com/JuliaArrays/MappedArrays.jl) to emulate the
  old behaviour.
* Same comments for `pyconvert(PyArray{String}, x)`.

## Julia wrappers (`JlDict`, etc.)

The wrapper types have been renamed.

* Instead of `juliacall.AnyValue` use `juliacall.Jl` (but see below).
* Instead of `juliacall.ArrayValue` use `juliacall.JlArray`.
* Instead of `juliacall.DictValue` use `juliacall.JlDict`.

Most methods on the `Jl` class return a `Jl` now instead of an arbitrary Python object
converted from the Julia return value. This makes generic programming easier and
more closely reflects the behaviour of `Py`.

* Instead of `jl.seval("1+2")` use `jl.jl_eval("1+2").jl_to_py()`.
* Instead of `jl.rand(5)[0]` use `jl.rand(5)[1].jl_to_py()`. Note the shift from 0-based
  to 1-based indexing - previously `jl.rand(5)` was a `juliacall.VectorValue` which
  supported Python 0-based indexing, but now `jl.rand(5)` is a `juliacall.Jl` which
  supports indexing by passing the arguments directly to Julia, which is 1-based.

Some wrapper types have been removed and can mostly be replaced with `Jl`.

* Instead of `juliacall.RawValue` use `juliacall.Jl`, since this behaves much the same
  now.
* Instead of `juliacall.IntegerValue` (and other number types) use `int`, `float`,
  `complex` or other numeric types as appropriate. Alternatively use `juliacall.Jl`
  which supports the basic arithmetic and comparison operators, but is not strictly a
  number.
* Instead of `juliacall.ModuleValue` use `juliacall.Jl`. The only benefit of
  `ModuleValue` was its `seval` method, which is now `Jl.jl_eval`.
* Instead of `juliacall.TypeValue` use `juliacall.Jl`. The only benefit of `TypeValue`
  was that indexing syntax (`jl.Vector[jl.Type]`) was converted to Julia's curly syntax
  (`Vector{Type}`) but `Jl` does this now (for types).

Methods with the `_jl_` prefix are renamed with the `jl_` prefix:
* Instead of `x._jl_help()` use `x.jl_help()`.
* Instead of `x._jl_display()` use `x.jl_display()`.

The `seval` function is now called `jl_eval`:
* Instead of `juliacall.Main.seval("1+2")` use `juliacall.Main.jl_eval("1+2")`.

Other methods, functions and attributes removed:
* Instead of `x._jl_raw()` use `x` (if already a `Jl`) or `Jl(x)`. This is because the
  old `AnyValue` and `RawValue` are replaced by `Jl`.
* Instead of `juliacall.convert(type, value)` use `juliacall.Jl(value, type)`.
* Instead of `juliacall.Pkg` you must import import it yourself, such as
  `juliacall.Main.jl_eval("using Pkg; Pkg")`.

On the Julia side, the `pyjl` function now always returns a `Jl`, whereas before it
would return one of the more specific wrappers (now called `JlDict`, `JlArray`, etc.).

* Instead of `pyjl([1, 2, 3])` use `pyjlarray([1, 2, 3])` if you need a `JlArray`.
* Instead of `pyjl(Dict())` use `pyjldict(Dict())` if you need a `JlDict`.
* Instead of `pyjl(Set())` use `pyjlset(Set())` if you need a `JlSet`.
* Continue to use `pyjl` if you are OK with the result being a `Jl`.
* Note that `Py([1, 2, 3])` still returns a `JlArray`, etc., only `pyjl` itself changed.
