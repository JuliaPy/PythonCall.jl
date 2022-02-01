# Coming from *PyCall*?

Another similar interface to Python is provided by [PyCall](https://github.com/JuliaPy/PyCall.jl).

On this page, we give some tips for migrating between the two modules and a comparison.

## Tips

- You can use both PyCall and PythonCall in the same Julia session (this might be platform dependent).
- To force PythonCall to use the same Python interpreter as PyCall, set the environment variable `JULIA_PYTHONCALL_EXE` to `"@PyCall"`.

## Comparison

### Flexibility of conversion

In PyCall you do `convert(T, x)` to convert the Python object `x` to a Julia `T`. In PythonCall you similarly do `pyconvert(T, x)`.

PythonCall supports far more combinations of types of `T` and `x`. For example `convert(Vector, x)` in PyCall requires `x` to be a sequence, whereas in PythonCall `pyconvert(Vector, x)` works if `x` is an iterable, an object supporting the buffer protocol (such as `bytes`) or an object supporting the numpy array interface (such as `numpy.ndarray`).

Furthermore, `pyconvert` can be extended to support more types, whereas `convert(Vector, x)` cannot support more Python types.

### Lossiness of conversion

Both packages allow conversion of Julia values to Python: `PyObject(x)` in PyCall, `Py(x)` in PythonCall.

Whereas both packages convert numbers, booleans, tuples and strings to their Python counterparts, they differ in handling other types. For example PyCall converts `AbstractVector` to `list` whereas PythonCall converts `AbstractVector` to `juliacall.VectorValue` which is a sequence type directly wrapping the Julia value - this has the advantage that mutating the Python object also mutates the original Julia object.

Hence with PyCall the following does not mutate the original array `x`:
```julia
x = ["foo", "bar"]
PyObject(x).append("baz")
@show x # --> ["foo", "bar"]
```
whereas with PythonCall the following does mutate `x`:
```julia
x = ["foo", "bar"]
Py(x).append("baz")
@show x # --> ["foo", "bar", "baz"]
```

In fact, PythonCall has the policy that any mutable object will by default be wrapped in this way, which not only preserves mutability but makes conversion faster for large containers since it does not require taking a copy of all the data.

### Automatic conversion

In PyCall, most function calls, attribute accesses, indexing, etc. of Python object by default automatically convert their result to a Julia object. This means that the following
```julia
pyimport("sys").modules["KEY"] = "VALUE"
```
does not actually modify the modules dict because it was *copied* to a new Julia `Dict`. This was probably not intended, plus it wasted time copying the whole dictionary. Instead you must do
```julia
set!(pyimport(os)."environ", "KEY", "VALUE")
```

In PythonCall, we don't do any such automatic conversion: we always return `Py`. This means that the first piece of code above does what you think.

### Which Python

PyCall uses some global installation of Python - typically the version of Python installed on the system or used by Conda.

PythonCall uses a separate Conda environment for each Julia environment/project/package and installs Python (and other Python packages) into that. This means that different Julia projects can maintain an isolated set of Python dependencies (including the Python version itself).

### Corresponding Python packages

PyCall has the corresponding Python package [PyJulia](https://github.com/JuliaPy/pyjulia) for calling Julia from Python, and PythonCall similarly has JuliaCall.

One difference is between them is their code size: PyJulia is a large package, whereas JuliaCall is very small, with most of the implementation being in PythonCall itself. The practical up-shot is that PythonCall/JuliaCall have very symmetric interfaces; for example they use identical conversion policies and have the same set of wrapper types available.

Note also that JuliaCall will use a separate Julia project for each virtual/conda environment. This means that different Python environments can maintain an isolated set of Julia dependencies, including the versions of Julia and PythonCall themselves.

### Compatibility

PyCall supports Julia 0.7+ and Python 2.7+, whereas PythonCall supports Julia 1.4+ and Python 3.5+. PyCall requires numpy to be installed, PythonCall doesn't (it provides the same fast array access through the buffer protocol and array interface).
