# The Julia module PythonCall

## Installation

This package is in the general registry, so to install just type `]` in the Julia REPL and run:

```julia-repl
pkg> add PythonCall
```

## Getting started

Import the module with:

```julia-repl
julia> using PythonCall
```

By default this will initialize a conda environment in your Julia environment, install
Python into it, load the corresponding Python library and initialize an interpreter. See
[here](@ref pythoncall-config) to configure which Python to use.

Now you can interact with Python as follows:

```julia-repl
julia> re = pyimport("re")
Python: <module 're' from '[...]/lib/re.py'>

julia> words = re.findall("[a-zA-Z]+", "PythonCall.jl is very useful!")
Python: ['PythonCall', 'jl', 'is', 'very', 'useful']

julia> sentence = Py(" ").join(words)
Python: 'PythonCall jl is very useful'

julia> pyconvert(String, sentence)
"PythonCall jl is very useful"
```

In this example:
- We used [`pyimport`](@ref) to import the `re` module.
- We called its `findall` function on a pair of strings, which were automatically
  converted to Python strings (see [Conversion to Python](@ref jl2py)).
- We called [`Py`](@ref) to explicitly convert a string to a Python string, so that we
  could call its `join` method. All Python objects are of type `Py`.
- We called [`pyconvert`](@ref) to convert the Python string `sentence` to a Julia string
  (see [Conversion to Julia](@ref py2jl)).

The values `re`, `words` and `sentence` in the example are all Python objects, which have
type [`Py`](@ref) in Julia. As we have seen, these objects support attribute access (e.g.
`re.findall`) and function calls (e.g. `join(words)`). They also support indexing,
comparison and arithmetic:

```julia-repl
julia> x = pylist([3, 4, 5])
Python: [3, 4, 5]

julia> x[2] == 5
Python: True

julia> x[pyslice(0,2)] + pylist([1,2])
Python: [3, 4, 1, 2]
```

We have just seen the functions [`pylist`](@ref) (for constructing a Python list) and
[`pyslice`](@ref) (for constructing a Python slice). There are many such functions,
mirroring most of the Python builtin functions and types. The
[API Reference](@ref py-reference) documents them all.

Most of these functions are essentially Python builtins with a `py` prefix. For example
`pyint(x)` converts `x` to a Python `int` and is equivalent to `int(x)` in Python when `x`
is a Python object.

Notable exceptions are:
- [`pyconvert`](@ref) to convert a Python object to a Julia object.
- [`pyimport`](@ref) to import a Python module.
- [`pyjl`](@ref) to directly wrap a Julia object as a Python object.
- [`pywith`](@ref) to emulate the Python `with` statement.

To access the Python builtins directly, you can access the fields of [`pybuiltins`](@ref):
```julia-repl
julia> pybuiltins.None
Python: None

julia> pybuiltins.True
Python: True

julia> pybuiltins.ValueError("some error")
Python: ValueError('some error')
```

With the functions introduced so far, you have access to the vast majority of Python's
functionality.

## Conversion between Julia and Python

A Julia object can be converted to a Python one either explicitly (such as `Py(x)`) or
implicitly (such as the arguments when calling a Python function). Either way, it follows
the default conversion rules [here](@ref jl2py).

Most operations involving Python objects will return a `Py` and are not automatically
converted to another Julia type. Instead, you can explicitly convert using
[`pyconvert`](@ref):

```julia-repl
julia> x = pylist([3.4, 5.6])
Python: [3.4, 5.6]

julia> pyconvert(Vector, x)
2-element Vector{Float64}:
 3.4
 5.6

julia> pyconvert(Vector{Float32}, x)
2-element Vector{Float32}:
 3.4
 5.6

julia> pyconvert(Any, x)
2-element PyList{Any}:
 3.4
 5.6
```

In the above example, we converted a Python list to a Julia vector in three ways.
- `pyconvert(Vector, x)` returned a `Vector{Float64}` since all the list items are floats.
- `pyconvert(Vector{Float32}, x)` specified the element type, so the floats were converted
  to `Float32`.
- `pyconvert(Any, x)` returned a `PyList{Py}` which is a no-copy wrapper around the original
  list `x`, viewing it as a `AbstractVector{Py}`. Since it is a wrapper, mutating it
  mutates `x` and vice-versa.

See [here](@ref py2jl) for the rules regarding how `pyconvert(T, x)` works. If `x` is an
immutable scalar type (such as an `int` or `str`) then `pyconvert(Any, x)` may return the
corresponding Julia object (such as an `Integer` or `String`). Otherwise it will typically
return either a [wrapper type](@ref py-wrappers) (such as `PyList{Py}` in the above
example) or will fall back to returning a [`Py`](@ref).

## [Wrapper types](@id py-wrappers)

A wrapper is a type which wraps a Python object but provides it with the semantics of some
other Julia type.

Since it is merely wrapping a Python object, if you mutate the wrapper you also mutate the
wrapped object, and vice versa.

See [here](@ref python-wrappers) for details of all the wrapper types provided by
PythonCall.

We have already seen [`PyList`](@ref). It wraps any Python sequence (such as a list) as
a Julia vector:

```julia-repl
julia> x = pylist([3,4,5])
Python: [3, 4, 5]

julia> y = PyList{Union{Int,Nothing}}(x)
3-element PyList{Union{Nothing, Int64}}:
 3
 4
 5

julia> push!(y, nothing)
4-element PyList{Union{Nothing, Int64}}:
 3
 4
 5
  nothing

julia> append!(y, 1:2)
6-element PyList{Union{Nothing, Int64}}:
 3
 4
 5
  nothing
 1
 2

julia> x
Python: [3, 4, 5, None, 1, 2]
```

There are wrappers for other container types, such as [`PyDict`](@ref) and [`PySet`](@ref).

The wrapper [`PyArray`](@ref) provides a Julia array view of any Python array, i.e. anything
satisfying either the buffer protocol or the numpy array interface. This includes things
like `bytes`, `bytearray`, `array.array` and `numpy.ndarray`:

```julia-repl
julia> x = pyimport("array").array("i", [3, 4, 5])
Python: array('i', [3, 4, 5])

julia> y = PyArray(x)
3-element PyArray{Int32, 1, true, true, Int32}:
 3
 4
 5

julia> sum(y)
12

julia> y[1] = 0
0

julia> x
Python: array('i', [0, 4, 5])
```

It directly wraps the underlying data buffer, so array operations such as indexing are about
as fast as for an ordinary `Array`.

The [`PyIO`](@ref) wrapper type views a Python file object as a Julia IO object:

```julia-repl
julia> x = pyimport("io").StringIO()
Python: <_io.StringIO object at 0x000000006579BC70>

julia> y = PyIO(x)
PyIO(<py _io.StringIO object at 0x000000006579BC70>, false, true, false, 4096, UInt8[], 4096, UInt8[])

julia> println(y, "Hello, world!")

julia> flush(y)

julia> x.seek(0)
Python: 0

julia> x.read()
Python: 'Hello, world!\n'
```

## [Configuration](@id pythoncall-config)

By default, PythonCall uses [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl) to manage
its dependencies. This will install Conda and use it to create a Conda environment specific
to your current Julia project containing Python and any required Python packages.

#### If you already have Python and required Python packages installed

```julia
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/path/to/python"  # optional
ENV["JULIA_PYTHONCALL_EXE"] = "@PyCall"  # optional
```

By setting the CondaPkg backend to Null, it will never install any Conda packages. In this
case, PythonCall will use whichever Python is currently installed and in your `PATH`. You
must have already installed any Python packages that you need.

If `python` is not in your `PATH`, you will also need to set `JULIA_PYTHONCALL_EXE` to its
path.

If you also use PyCall, you can set `JULIA_PYTHONCALL_EXE=@PyCall` to use the same Python
interpreter.

#### If you already have a Conda environment

```julia
ENV["JULIA_CONDAPKG_BACKEND"] = "Current"
ENV["JULIA_CONDAPKG_EXE"] = "/path/to/conda"  # optional
```

The Current backend to CondaPkg will use the currently activated Conda environment instead
of creating a new one.

Note that this will still install any required Conda packages into your Conda environment.
If you already have your dependencies installed and do not want the environment to be
modified, then see the previous section.

If `conda`, `mamba` or `micromamba` is not in your `PATH` you will also need to set
`JULIA_CONDAPKG_EXE` to its path.

#### If you already have Conda, Mamba or MicroMamba

```julia
ENV["JULIA_CONDAPKG_BACKEND"] = "System"
ENV["JULIA_CONDAPKG_EXE"] = "/path/to/conda"  # optional
```

The System backend to CondaPkg will use your preinstalled Conda implementation instead of
downloading one.

Note that this will still create a new Conda environment and install any required packages
into it. If you want to use a pre-existing Conda environment, see the previous section.

If `conda`, `mamba` or `micromamba` is not in your `PATH` you will also need to set
`JULIA_CONDAPKG_EXE` to its path.

## [Installing Python packages](@id python-deps)

Assuming you haven't [opted out](@ref pythoncall-config), PythonCall uses
[CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl) to automatically install any required
Python packages.

This is as simple as
```julia-repl
julia> using CondaPkg

julia> # press ] to enter the Pkg REPL

pkg> conda add some_package
```

This creates a `CondaPkg.toml` file in the active project specifying the dependencies, just
like a `Project.toml` specifies Julia dependencies. Commit this file along with the rest of
the project so that dependencies are automatically installed for everyone using it.

To add dependencies to a Julia package, just ensure the package project is activated first.

See the [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl) documentation.

## Writing packages which depend on PythonCall

### Example

See [https://github.com/cjdoris/Faiss.jl](https://github.com/cjdoris/Faiss.jl) for an
example package which wraps the Python FAISS package.

### Precompilation

You must not interact with Python during module precompilation. Therefore, instead of
```julia
module MyModule
  using PythonCall
  const foo = pyimport("foo")
  bar() = foo.bar() # will crash when called
end
```
you can do the import when the module is loaded, saving the result in a `Ref`
```julia
module MyModule
  using PythonCall
  const foo = Ref{Py}()
  function __init__()
    foo[] = pyimport("foo")
  end
  bar() = foo[].bar()
end
```
or you can perform any imports dynamically
```julia
module MyModule
  using PythonCall
  bar() = pyimport("foo").bar()
end
```
or if that is too slow, you can cache the import
```julia
module MyModule
  using PythonCall
  bar() = @pyconst(pyimport("foo")).bar()
end
```
or even cache the imported function
```julia
module MyModule
  using PythonCall
  bar() = @pyconst(pyimport("foo").bar)()
end
```

### Dependencies

If your package depends on some Python packages, you must generate a `CondaPkg.toml` file.
See [Installing Python packages](@ref python-deps).

## [Multi-threading](@id jl-multi-threading)

From v0.9.22, PythonCall supports multi-threading in Julia and/or Python, with some
caveats.

Most importantly, you can only call Python code while Python's
[Global Interpreter Lock (GIL)](https://docs.python.org/3/glossary.html#term-global-interpreter-lock)
is locked by the current thread. Ordinarily, the GIL is locked by the main thread in Julia,
so if you want to run Python code on any other thread, you must unlock the GIL from the
main thread and then re-lock it while running any Python code on other threads.

This is made possible by the macros [`PythonCall.GIL.@unlock`](@ref) and
[`PythonCall.GIL.@lock`](@ref) or the functions [`PythonCall.GIL.unlock`](@ref) and
[`PythonCall.GIL.lock`](@ref) with this pattern:

```julia
PythonCall.GIL.@unlock Threads.@threads for i in 1:4
  PythonCall.GIL.@lock pyimport("time").sleep(5)
end
```

In the above example, we call `time.sleep(5)` four times in parallel. If Julia was
started with at least four threads (`julia -t4`) then the above code will take about
5 seconds.

Both `@unlock` and `@lock` are important. If the GIL were not unlocked, then a deadlock
would occur when attempting to lock the already-locked GIL from the threads. If the GIL
were not re-locked, then Python would crash when interacting with it.

You can also use [multi-threading from Python](@ref py-multi-threading).

### Caveat: Garbage collection

If Julia's GC collects any Python objects from a thread where the GIL is not currently
locked, then those Python objects will not immediately be deleted. Instead they will be
queued to be deleted in a later GC pass.

If you find you have many Python objects not being deleted, you can call
[`PythonCall.GC.gc()`](@ref) or `GC.gc()` while the GIL is locked to clear the queue.
