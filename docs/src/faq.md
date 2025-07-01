# FAQ & Troubleshooting

## Can I use PythonCall and PyCall together?

Yes, you can use both PyCall and PythonCall in the same Julia session. This is platform-dependent:
- On most systems the Python interpreter used by PythonCall and PyCall must be the same (see below).
- On Windows it appears to be possible for PythonCall and PyCall to use different interpreters.

To force PythonCall to use the same Python interpreter as PyCall, set the environment variable [`JULIA_PYTHONCALL_EXE`](@ref pythoncall-config) to `"@PyCall"`. Note that this will opt out of automatic dependency management using CondaPkg.

Alternatively, to force PyCall to use the same interpreter as PythonCall, set the environment variable `PYTHON` to [`PythonCall.python_executable_path()`](@ref) and then `Pkg.build("PyCall")`. You will need to do this each time you change project, because PythonCall by default uses a different Python for each project.

## Is PythonCall/JuliaCall thread safe?

Yes, as of v0.9.22, provided you handle the GIL correctly. See the guides for
[PythonCall](@ref jl-multi-threading) and [JuliaCall](@ref py-multi-threading).

Before, tricks such as disabling the garbage collector were required. See the
[old docs](https://juliapy.github.io/PythonCall.jl/v0.9.21/faq/#Is-PythonCall/JuliaCall-thread-safe?).

Related issues:
[#201](https://github.com/JuliaPy/PythonCall.jl/issues/201),
[#202](https://github.com/JuliaPy/PythonCall.jl/issues/202),
[#529](https://github.com/JuliaPy/PythonCall.jl/pull/529)

## Issues when Numpy arrays are expected

When a Julia array is passed to Python, it is wrapped as a [`ArrayValue`](#juliacall.ArrayValue).
This type satisfies the Numpy array interface and the buffer protocol, so can be used in
most places where a numpy array is valid.

However, some Python packages have functions which assume the input is an actual Numpy array.
You may see errors such as:
```
AttributeError: Julia: type Array has no field dtype
```

To fix this you can convert the array `x` to a Numpy array as follows
```julia
Py(x).to_numpy()
```

If the array is being mutated, you will need to pass the argument `copy=false`.

Related issues: [#280](https://github.com/JuliaPy/PythonCall.jl/issues/280)

## Heap corruption when using PyTorch

On some systems, you may see an error like the following when using `torch` and `juliacall`:
```text
Python(65251,0x104cf8580) malloc: Heap corruption detected, free list is damaged at 0x600001c17280
*** Incorrect guard value: 1903002876
Python(65251,0x104cf8580) malloc: *** set a breakpoint in malloc_error_break to debug
[1]    65251 abort      ipython
```

A solution is to ensure that `juliacall` is imported before `torch`.

Related issues: [#215](https://github.com/JuliaPy/PythonCall.jl/issues/215)

## `ccall requires the compiler` error when importing some Python libraries
On some systems, you may see an error like the following when import e.g. `matplotlib` before `juliacall`:

```
ERROR: `ccall` requires the compilerTraceback (most recent call last):
  File "/home/dingraha/projects/pythoncall_import_error/run.py", line 2, in <module>
    from foo import Foo
  File "/home/dingraha/projects/pythoncall_import_error/foo.py", line 4, in <module>
    import juliacall; jl = juliacall.newmodule("FooModule")
  File "/home/dingraha/projects/pythoncall_import_error/venv/lib/python3.9/site-packages/juliacall/__init__.py", line 218, in <module>
    init()
  File "/home/dingraha/projects/pythoncall_import_error/venv/lib/python3.9/site-packages/juliacall/__init__.py", line 214, in init
    raise Exception('PythonCall.jl did not start properly')
Exception: PythonCall.jl did not start properly
```

The likely problem is that the "other" Python library (`matplotlib`, whatever) is loading the system `libstdc++.so`, which isn't compatible with the `libstdc++.so` that Julia ships with.
Linux distributions with older `libstdc++` versions seem more likely to suffer from this issue.
The solution is to either:

  * use a Linux distribution with a more recent `libstdc++`
  * import `juliacall` before the other Python library, so that Julia's `libstdc++` is loaded
  * use a Python from a conda environment, which will have a newer `libstdc++` that is compatible with Julia's

Related issues: [#255](https://github.com/JuliaPy/PythonCall.jl/issues/255)

## Can I use JuliaCall to run Julia inside applications with embedded Python?

Yes, it may be possible. A good example of that is having Julia running inside the Python that is running inside Blender, as presented in [this Discourse post](https://discourse.julialang.org/t/running-julia-inside-blender-through-vscode-using-pythoncall-juliacall/96838/6).
From the point that one has JuliaCall running inside Python, if it has access to the terminal, one can even launch a Julia REPL there, and if needed connect with VSCode Julia extension to it.
The full Python script to install, launch JuliaCall, and launch a Julia REPL in Blender is [here](https://gist.github.com/cdsousa/d820d27174238c0d48e5252355584172).

## Using PythonCall.jl and CondaPkg.jl in a script

If running from a script, make sure that [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl) is used before [PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl) to ensure proper loading of Python packages in your path. E.g.,

```julia
using CondaPkg

CondaPkg.add("numpy")

using PythonCall

np = pyimport("numpy")
```
