# Release Notes

## Unreleased (v1)
* `juliacall.Pkg` is removed.
* The following functionality has been moved into package extensions, and therefore is now
  only available on Julia 1.9+:
  * TODO

## Unreleased
* `Py` is now treated as a scalar when broadcasting.
* Bug fixes.

## 0.9.15 (2023-10-25)
* JuliaCall now supports `-X juliacall-startup-file=no` to disable running startup.jl.
* If you are using CondaPkg then Python can optionally now be installed from the anaconda
  channel (instead of only conda-forge).
* Bug fixes.

## 0.9.14 (2023-07-26)
* Wrapped Julia values support truthiness (`__bool__`) better: all values are true, except
  for zero numbers and empty arrays, dicts and sets.
* JuliaCall now supports the Julia `--handle-signals` option. Setting this to `yes` allows
  allocating multithreaded Julia code to be called from JuliaCall without segfaulting. The
  default is `no` while compatibility concerns are investigated, and may be changed to `yes`
  in a future release.

## 0.9.13 (2023-05-14)
* Conversion to wrapper types `PyList`, `PySet`, `PyDict` or `PyIterable` now default to
  having element type `Any` instead of `Py`.
* The `__repr__` method of wrapped Julia objects now uses the 3-arg show method for nicer
  (richer and truncated) display at the Python REPL.
* The IPython extension can now be loaded as just `%load_ext juliacall`.
* The `%%julia` IPython magic can now synchronise variables between Python and Julia.
* Bug fixes.

## 0.9.12 (2023-02-28)
* Bug fixes.

## 0.9.11 (2023-02-15)
* In `PyArray{T}(x)`, the eltype `T` no longer needs to exactly match the stored data type.
  If `x` has numeric elements, then any number type `T` is allowed. If `x` has string
  elements, then any string type `T` is allowed.
* `StaticString` (the inline string type used by `PyArray`) supports the `AbstractString`
  interface better.

## 0.9.10 (2022-12-02)
* Bug fixes.

## 0.9.9 (2022-10-20)
* Bug fixes.

## 0.9.8 (2022-10-18)
* Adds `line_buffering` option to `PyIO`.
* Improvements to stdout when using `juliacall.ipython` including line-buffering.

## 0.9.7 (2022-10-11)
* If CondaPkg is using the Null backend, PythonCall will now use `python` from the PATH.
* Bug fixes.

## 0.9.6 (2022-09-09)
* When using JuliaCall from an interactive Python session, Julia is put into interactive
  mode: `isinteractive()` is true, InteractiveUtils is loaded, and a nicer display is used.
* Wrapped Julia values now truncate their output when displayed via `_repr_mimebundle_`.
* Numpy arrays with structured dtypes can now be converted to `PyArray`, provided the fields
  are aligned.
* Python named tuples can be converted to Julia named tuples.
* Bug fixes.

## 0.9.5 (2022-08-19)
* Adds `PythonCall.GC.disable()` and `PythonCall.GC.enable()`.
* Experimental new function `juliacall.interactive()` allows the Julia async event loop to
  run in the background of the Python REPL.
* Experimental new IPython extension `juliacall.ipython` providing the `%julia` and `%%julia`
  magics for executing Julia code.
* Experimental new module `juliacall.importer` allowing you to write Python modules in
  Julia.
* Bug fixes.

## 0.9.4 (2022-07-26)
* Bug fixes.

## 0.9.3 (2022-07-02)
* Bug fixes.

## 0.9.2 (2022-07-02)
* Many Julia CLI options (such sysimage or number of threads) can be set from JuliaCall.
* Bug fixes.

## 0.9.1 (2022-06-18)
* `PyArray` can be constructed using the `__array_struct__` part of the Numpy array
  interface. Constructing `PyArray(x)` is now about 50x faster, or 175x faster if you fully
  specify the type.
* JuliaCall can now be imported on Apple M1.

## 0.9.0 (2022-05-27)
* **Breaking.** Removes `getpy`: you may now overload `Py` directly, which now need not
  always return a new object (e.g. for singletons or wrappers).
* **Breaking.** Conversion rules no longer take a new object every time.
* **Breaking.** Improved Tables-interface support for `PyPandasDataFrame`: better inferred
  column types; better handling of non-string column names; columns are usually wrappers
  (`PyArray` or `PyList`). Constructor arguments have changed. Dict methods have been
  removed (basically only the Tables interface is supported).
* **Breaking.** A `Py` which is convertible to `PyTable` is no longer considered to be a
  table itself; you must convert explicitly.
* Adds `pyhasitem` and 3-arg `pygetitem`.
* Extends `Base.get`, `Base.get!`, `Base.haskey` and 2-arg `Base.hash` for `Py`.
* `PyArray` can now have any element type when the underlying array is of Python objects.
* Adds `ArrayValue.to_numpy()`.
* Bug fixes.

## v0.8.0 (2022-03-17)
* **Breaking:** Removes `pymethod` and `pyclass`. In the future, `pyclass` may become sugar
  for `types.new_class` (namely you can specify a metaclass).
* Adds `pyfunc`, `pyclassmethod`, `pystaticmethod` and `pyproperty`.
* `pyconvert_add_rule` is now documented. Its semantics have changed, including the
  separator of the first argument from `/` to `:`.
* A pandas `<NA>` value is now converted to `missing`.
* A `NaN` in a `PyPandasDataFrame` is converted to `missing`.
* **Breaking:** Removes `using` and `As` from JuliaCall.
* Adds `convert` to JuliaCall (replacing `As`).
* Bug fixes.

## v0.6.1 (2022-02-21)
* Conversions from simple ctypes types, e.g. `ctypes.c_float` to `Cfloat`.
* Conversions from simple numpy types, e.g. `numpy.float32` to `Float32`.
* Bug fixes.

## v0.6.0 (2022-02-17)
* **Breaking:** JuliaCall now uses JuliaPkg to manage Julia dependencies.
* Bug fixes.

## v0.5.1 (2022-01-24)
* Bug fixes.

## v0.5.0 (2021-12-11)
* **Breaking:** PythonCall now uses CondaPkg to manage Python dependencies.
* Python objects can be shared with PyCall provided it uses the same interpreter, using methods `PythonCall.Py(::PyCall.PyObject)` and `PyCall.PyObject(::PythonCall.Py)`.
* Adds `PythonDisplay` which displays objects by printing to Python's `sys.stdout`. Used automatically in IPython in addition to `IPythonDisplay`.
* Removes the `GLOBAL` mode from `@pyexec`. Use `global` in the code instead.
* Bug fixes.
