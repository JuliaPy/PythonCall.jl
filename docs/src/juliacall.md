# The Python module JuliaCall

## Installation

It's as simple as
```bash
pip install juliacall
```

Developers may wish to clone the repo (https://github.com/JuliaPy/PythonCall.jl) directly
and pip install the module in editable mode. You should add `"dev":true, "path":"../.."` to
`python/juliacall/juliapkg.json` to ensure you use the development version of PythonCall
in conjunction with JuliaCall.

## Getting started

For interactive or scripting use, the simplest way to get started is:
```python
from juliacall import Main as jl
```

This loads a single variable `jl` which represents the `Main` module in Julia,
from which all of Julia's functionality is available:
```python
jl.println("Hello from Julia!")
# Hello from Julia!
x = jl.rand(range(10), 3, 5)
x._jl_display()
# 3×5 Matrix{Int64}:
#  8  1  7  0  6
#  9  2  1  4  0
#  1  8  5  4  0
import numpy
numpy.sum(x, axis=0)
# array([18, 11, 13,  8,  6], dtype=int64)
```

In this example:
- We called the `jl.println` function to print a message.
- We called the `jl.rand` function to generate an array of random integers. Note that the
  first argument is `range(10)` which is converted to `0:9` in Julia.
- We called its special `_jl_display()` to show it using Julia's display mechanism.
- We called the `numpy.sum` function to sum each column of `x`. This automatically converted
  `x` to a NumPy array. (We could have done `jl.sum(x, dims=1)` too.)

If you are writing a package which uses Julia, then to avoid polluting the global `Main`
namespace you instead should start with:
```python
import juliacall
jl = juliacall.newmodule("SomeName")
```

Julia modules have a special method `seval` which will evaluate a given piece of code given
as a string in the module. This is most frequently used to import modules:
```python
from array import array
jl.seval("using Statistics")
x = array('i', [1, 2, 3])
jl.mean(x)
# 2.0
y = array('i', [2,4,8])
jl.cor(x, y)
# 0.9819805060619657
```

What to read next:
- The main functionality of this package is in `AnyValue` objects, which represent Julia
  objects, [documented here](@ref julia-wrappers).
- If you need to install Julia packages, [read here](@ref julia-deps).
- When you call a Julia function, such as `jl.rand(...)` in the above example, its
  arguments are converted to Julia according to [this table](@ref py2jl-conversion) and
  its return value is converted to Python according to [this table](@ref jl2py-conversion).

## [Managing Julia dependencies](@id julia-deps)

JuliaCall manages its Julia dependencies using [JuliaPkg](https://github.com/JuliaPy/PyJuliaPkg).

It will automatically download a suitable version of Julia if required.

A Julia environment is also created, activated and populated with any required packages.
If you are in a virtual or Conda environment, the environment is put there. Otherwise a
global environment is used at `~/.julia/environments/pyjuliapkg`.

If your project requires any Julia packages, or a particular version of Julia itself, then
create a file called `juliapkg.json` in your package. For example:
Here is an example:
```json
{
    "julia": "1.5",
    "packages": {
        "Example": {
            "uuid": "7876af07-990d-54b4-ab0e-23690620f79a",
            "version": "0.5, 0.6"
        }
    }
}
```

Alternatively you can use `add`, `rm`, etc. from JuliaPkg to edit this file.

See [JuliaPkg](https://github.com/JuliaPy/PyJuliaPkg) for more details.

## [Configuration](@id julia-config)

Some features of the Julia process, such as the optimization level or number of threads, may
be configured in two ways:
- As an `-X` argument to Python, such as `-X juliacall-optlevel=3`; or
- As an environment variable, such as `PYTHON_JULIACALL_OPTLEVEL=3`.

| `-X` option | Environment Variable | Description |
| :---------- | :------------------- | :---------- |
| `-X juliacall-home=<dir>` | `PYTHON_JULIACALL_BINDIR=<dir>` | The directory containing the julia executable. |
| `-X juliacall-check-bounds=<yes\|no\|auto>` | `PYTHON_JULIACALL_CHECK_BOUNDS=<yes\|no\|auto>` | Enable or disable bounds checking. |
| `-X juliacall-compile=<yes\|no\|all\|min>` | `PYTHON_JULIACALL_COMPILE=<yes\|no\|all\|min>` | Enable or disable JIT compilation. |
| `-X juliacall-compiled-modules=<yes\|no>` | `PYTHON_JULIACALL_COMPILED_MODULES=<yes\|no>` | Enable or disable incrementally compiling modules. |
| `-X juliacall-depwarn=<yes\|no\|error>` | `PYTHON_JULIACALL_DEPWARN=<yes\|no\|error>` | Enable or disable deprecation warnings. |
| `-X juliacall-handle-signals=<yes\|no>` | `PYTHON_JULIACALL_HANDLE_SIGNALS=<yes\|no>` | Enable or disable Julia signal handling. |
| `-X juliacall-inline=<yes\|no>` | `PYTHON_JULIACALL_INLINE=<yes\|no>` | Enable or disable inlining. |
| `-X juliacall-min-optlevel=<0\|1\|2\|3>` | `PYTHON_JULIACALL_MIN_OPTLEVEL=<0\|1\|2\|3>` | Optimization level. |
| `-X juliacall-optimize=<0\|1\|2\|3>` | `PYTHON_JULIACALL_OPTIMIZE=<0\|1\|2\|3>` | Minimum optimization level. |
| `-X juliacall-procs=<N\|auto>` | `PYTHON_JULIACALL_PROCS=<N\|auto>` | Launch N local worker process. |
| `-X juliacall-startup-file=<yes\|no>` | `PYTHON_JULIACALL_STARTUP_FILE=<yes|no>` | Enable or disable your startup.jl file. |
| `-X juliacall-sysimage=<file>` | `PYTHON_JULIACALL_SYSIMAGE=<file>` | Use the given system image. |
| `-X juliacall-threads=<N\|auto>` | `PYTHON_JULIACALL_THREADS=<N\|auto>` | Launch N threads. |
| `-X juliacall-warn-overwrite=<yes\|no>` | `PYTHON_JULIACALL_WARN_OVERWRITE=<yes\|no>` | Enable or disable method overwrite warnings. |
