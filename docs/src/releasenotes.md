# Release Notes

## Unreleased
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
