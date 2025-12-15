# Changelog

## Unreleased (v1)
* The vast majority of these changes are breaking, see the [v1 Migration Guide](@ref) for how to upgrade.
* Changes to core functionality:
  * Comparisons like `==(::Py, ::Py)`, `<(::Py, ::Number)`, `isless(::Number, ::Py)` now return `Bool` instead of `Py`.
* Changes to `PythonCall.GC` (now more like `Base.GC`):
  * `enable(true)` replaces `enable()`.
  * `enable(false)` replaces `disable()`.
  * `gc()` added.
* Changes to Python wrapper types:
  * `PyArray` has been reparametrised from `PyArray{T,N,M,L,R}` to `PyArray{T,N,F}`:
    * `F` is a tuple of symbols representing flags, with `:linear` replacing `L` and `:mutable` replacing `M`.
    * `R` is removed and is now implied by `T`, which currently must be either a bits type (equal to `R`) or `Py`, or a tuple of these.
* Changes to Julia wrapper types:
  * Classes renamed: `ValueBase` to `JlBase`, `AnyValue` to `Jl`, `ArrayValue` to `JlArray`, etc.
  * Classes removed: `RawValue`, `ModuleValue`, `TypeValue`, `NumberValue`, `ComplexValue`, `RealValue`, `RationalValue`, `IntegerValue`.
  * `Jl` now behaves similar to how `RawValue` behaved before. In particular, most methods on `Jl` now return a `Jl` instead of an arbitrary Python object.
  * `juliacall.Pkg` removed (you can import it yourself).
  * `juliacall.convert` removed (use `juliacall.Jl` instead).
  * Methods renamed: `_jl_display()` to `jl_display()`, `_jl_help()` to `jl_help()`, etc.
  * Methods removed: `_jl_raw()`.
  * `pyjl(x)` now always returns a `juliacall.Jl` (it used to select a wrapper type if possible).
  * `pyjltype(x)` removed.
  * New functions: `pyjlarray`, `pyjldict`, `pyjlset`.

## 0.9.30 (2025-11-18)
* Maximum supported Python version is now 3.13 ([see the FAQ](https://juliapy.github.io/PythonCall.jl/stable/faq/#faq-python-314) for why).

## 0.9.29 (2025-11-14)
* Minimum supported Python version is now 3.10.
* Minimum supported Julia version is now 1.10.
* Showing `Py` now respects the `compact` option - output is limited to a single line of
  at most the display width.
* Support policy now documented in the FAQ.
* Added this changelog (was previously at `docs/src/releasenotes.md`).
* Bug fixes.

## 0.9.28 (2025-09-17)
* Added `NumpyDates`: NumPy-compatible DateTime64/TimeDelta64 types and units.
* Added `pyconvert` rules for NumpyDates types.
* Added `PyArray` support for NumPy arrays of `datetime64` and `timedelta64`.
* Added `juliacall.ArrayValue` support for Julia arrays of `InlineDateTime64` and `InlineTimeDelta64`.
* If `JULIA_PYTHONCALL_EXE` is a relative path, it is now considered relative to the active project.
* Added option `JULIA_PYTHONCALL_EXE=@venv` to use a Python virtual environment relative to the active project.
* Added `PYTHON_JULIACALL_EXE` and `PYTHON_JULIACALL_PROJECT` for specifying the Julia binary and project to override JuliaPkg.
* Adds methods `Py(::AbstractString)`, `Py(::AbstractChar)` (previously only builtin string and char types were allowed).
* Adds methods `Py(::Integer)`, `Py(::Rational{<:Integer})`, `Py(::AbstractRange{<:Integer})` (previously only builtin integer types were allowed).
* Adds method `pydict(::Pair...)` to construct a python `dict` from `Pair`s, similar to `Dict`.
* Bug fixes.
* Internal: switch from Requires.jl to package extensions.

## 0.9.27 (2025-08-19)
* Internal: Use heap-allocated types (PyType_FromSpec) to improve ABI compatibility.
* Minimum supported Python version is now 3.9.
* Better compatibility with libstdc++.

## 0.9.26 (2025-07-15)
* Added PySide6 support to the GUI compatibility layer.
* Added FAQ on interactive threads.
* Added CI benchmarking suite.
* Bug fixes.

## 0.9.25 (2025-05-13)
* Added `PYTHON_JULIACALL_HEAP_SIZE_HINT` option to configure initial Julia heap size.
* `Base.elsize` now defined for `PyArray`.
* JuliaCall now ensures a version of OpenSSL_jll compatible with Python is installed.

## 0.9.24 (2025-01-22)
* Bug fixes.

## 0.9.23 (2024-08-22)
* Bug fixes.

## 0.9.22 (2024-08-07)
* Finalizers are now thread-safe, meaning PythonCall now works in the presence of
  multi-threaded Julia code. Previously, tricks such as disabling the garbage collector
  were required. Python code must still be called on the main thread.
* `GC.disable()` and `GC.enable()` are now a no-op and deprecated since they are no
  longer required for thread-safety. These will be removed in v1.
* Adds `GC.gc()`.
* Adds module `GIL` with `lock()`, `unlock()`, `@lock` and `@unlock` for handling the
  Python Global Interpreter Lock. In combination with the above improvements, these
  allow Julia and Python to co-operate on multiple threads.
* Adds method `_jl_call_nogil` to `juliacall.AnyValue` and `juliacall.RawValue` to call
  Julia functions with the GIL unlocked.

## 0.9.21 (2024-07-20)
* `Serialization.serialize` can use `dill` instead of `pickle` by setting the env var `JULIA_PYTHONCALL_PICKLE=dill`.
* `numpy.bool_` can now be converted to `Bool` and other number types.
* `datetime.timedelta` can now be converted to `Dates.Nanosecond`, `Microsecond`, `Millisecond` and `Second`. This behaviour was already documented.
* In JuliaCall, the Julia runtime is now properly terminated when Python exits. This means all finalizers should always run.
* NULL Python objects (such as from `pynew()`) can be safely displayed in multimedia contexts (VSCode/Pluto/etc.)

## 0.9.20 (2024-05-01)
* The IPython extension is now automatically loaded upon import if IPython is detected.
* JuliaCall now compatible with Julia 1.10.3.
* Minimum supported Python version is now 3.8.

## 0.9.19 (2024-03-19)
* Bug fixes.

## 0.9.18 (2024-03-18)
* Bug fixes.

## 0.9.17 (2024-03-16)
* Bug fixes.

## 0.9.16 (2024-03-14)
* Big internal refactor.
* New unexported functions: `python_executable_path`, `python_library_path`, `python_library_handle` and `python_version`.
* `Py` is now treated as a scalar when broadcasting.
* `PyArray` is now serializable.
* Removed compatibility with Julia 1.10.1 and 1.10.2 (to be fixed in 1.10.3 and 1.11.0) due to an upstream bug.
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
