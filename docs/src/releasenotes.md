# Release Notes

## Unreleased
* Adds `pyhasitem` and 3-arg `pygetitem`.
* Extends `Base.get`, `Base.get!`, `Base.haskey` and 2-arg `Base.hash` for `Py`.
* `PyArray` can now have any element type when the underlying array is of Python objects.
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
