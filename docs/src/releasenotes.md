# Release Notes

## Unreleased
* `pyconvert_add_rule` is now documented. The separator of the first argument has changed
  from `/` to `:` (not breaking because it was previously undocumented).

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
