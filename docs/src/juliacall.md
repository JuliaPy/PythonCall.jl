# The Python module JuliaCall

## Installation

It's as simple as
```bash
pip install juliacall
```

Developers may wish to clone the repo (https://github.com/JuliaPy/PythonCall.jl) directly
and pip install the module in editable mode. You should add `"dev":true, "path":"../.."` to
`pysrc/juliacall/juliapkg.json` to ensure you use the development version of PythonCall
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

By default JuliaCall manages its Julia dependencies using
[JuliaPkg](https://github.com/JuliaPy/PyJuliaPkg).

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

### Using existing environments

It's possible to override the defaults and disable JuliaPkg entirely by setting
the `PYTHON_JULIACALL_EXE` and `PYTHON_JULIACALL_PROJECT` options (both must be
set together). This is particularly useful when using shared environments on HPC
systems that may be readonly. Note that the project set in
`PYTHON_JULIACALL_PROJECT` *must* already have PythonCall.jl installed and it
*must* match the JuliaCall version, otherwise loading Julia will fail.

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
| `-X juliacall-startup-file=<yes\|no>` | `PYTHON_JULIACALL_STARTUP_FILE=<yes\|no>` | Enable or disable your startup.jl file. |
| `-X juliacall-sysimage=<file>` | `PYTHON_JULIACALL_SYSIMAGE=<file>` | Use the given system image. |
| `-X juliacall-threads=<N\|auto>` | `PYTHON_JULIACALL_THREADS=<N\|auto>` | Launch N threads. |
| `-X juliacall-warn-overwrite=<yes\|no>` | `PYTHON_JULIACALL_WARN_OVERWRITE=<yes\|no>` | Enable or disable method overwrite warnings. |
| `-X juliacall-autoload-ipython-extension=<yes\|no>` | `PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION=<yes\|no>` | Enable or disable IPython extension autoloading. |
| `-X juliacall-heap-size-hint=<N>` | `PYTHON_JULIACALL_HEAP_SIZE_HINT=<N>` | Hint for initial heap size in bytes. |
| `-X juliacall-exe=<file>` | `PYTHON_JULIACALL_EXE=<file>` | Path to Julia binary to use (overrides JuliaPkg). |
| `-X juliacall-project=<dir>` | `PYTHON_JULIACALL_PROJECT=<dir>` | Path to the Julia project to use (overrides JuliaPkg). |

## [Multi-threading](@id py-multi-threading)

!!! warning

    Multi-threading support is experimental and can change without notice.

From v0.9.22, JuliaCall supports multi-threading in Julia and/or Python, with some
caveats.

Most importantly, you can only call Python code while Python's
[Global Interpreter Lock (GIL)](https://docs.python.org/3/glossary.html#term-global-interpreter-lock)
is locked by the current thread. You can use JuliaCall from any Python thread, and the GIL
will be locked whenever any JuliaCall function is used. However, to leverage the benefits
of multi-threading, you can unlock the GIL while executing any Julia code that does not
interact with Python.

The simplest way to do this is using the `_jl_call_nogil` method on Julia functions to
call the function with the GIL unlocked.

```python
from concurrent.futures import ThreadPoolExecutor, wait
from juliacall import Main as jl
pool = ThreadPoolExecutor(4)
fs = [pool.submit(jl.Libc.systemsleep._jl_call_nogil, 5) for _ in range(4)]
wait(fs)
```

In the above example, we call `Libc.systemsleep(5)` on four threads. Because we
called it with `_jl_call_nogil`, the GIL was unlocked, allowing the threads to run in
parallel, taking about 5 seconds in total.

If we did not use `_jl_call_nogil` (i.e. if we did `pool.submit(jl.Libc.systemsleep, 5)`)
then the above code will take 20 seconds because the sleeps run one after another.

It is very important that any function called with `_jl_call_nogil` does not interact
with Python at all unless it re-locks the GIL first, such as by using
[PythonCall.GIL.@lock](@ref).

You can also use [multi-threading from Julia](@ref jl-multi-threading).

### Caveat: Julia's task scheduler

If you try the above example with a Julia function that yields to the task scheduler,
such as `sleep` instead of `Libc.systemsleep`, then you will likely experience a hang.

In this case, you need to yield back to Julia's scheduler periodically to allow the task
to continue. You can use the following pattern instead of `wait(fs)`:
```python
jl_yield = getattr(jl, "yield")
while True:
  # yield to Julia's task scheduler
  jl_yield()
  # wait for up to 0.1 seconds for the threads to finish
  state = wait(fs, timeout=0.1)
  # if they finished then stop otherwise try again
  if not state.not_done:
    break
```

Set the `timeout` parameter smaller to let Julia's scheduler cycle more frequently.

Future versions of JuliaCall may provide tooling to make this simpler.

### [Caveat: Signal handling](@id py-multi-threading-signal-handling)

We recommend setting [`PYTHON_JULIACALL_HANDLE_SIGNALS=yes`](@ref julia-config)
before importing JuliaCall with multiple threads.

This is because Julia intentionally causes segmentation faults as part of the GC
safepoint mechanism. If unhandled, these segfaults will result in termination of the
process. See discussion
[here](https://github.com/JuliaPy/PythonCall.jl/issues/219#issuecomment-1605087024)
for more information.

Note however that this interferes with Python's own signal handling, so for example
Ctrl-C will not raise `KeyboardInterrupt`.

Future versions of JuliaCall may make this the default behaviour when using multiple
threads.
