# Release Notes

## Unreleased
* Bug fixes.

## v0.5.1 (2022-01-24)
* Bug fixes.

## v0.5.0 (2021-12-11)
* **Breaking:** Now uses CondaPkg to manage Python dependencies.
* Python objects can be shared with PyCall provided it uses the same interpreter, using methods `PythonCall.Py(::PyCall.PyObject)` and `PyCall.PyObject(::PythonCall.Py)`.
* Adds `PythonDisplay` which displays objects by printing to Python's `sys.stdout`. Used automatically in IPython in addition to `IPythonDisplay`.
* Removes the `GLOBAL` mode from `@pyexec`. Use `global` in the code instead.
* Bug fixes.
