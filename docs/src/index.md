# PythonCall.jl

Bringing [**Python®**](https://www.python.org/) and [**Julia**](https://julialang.org/) together in seamless harmony:
- Call Python code from Julia and Julia code from Python via a symmetric interface.
- Simple syntax, so the Python code looks like Python and the Julia code looks like Julia.
- Intuitive and flexible conversions between Julia and Python: anything can be converted, you are in control.
- Fast non-copying conversion of numeric arrays in either direction: modify Python arrays (e.g. `bytes`, `array.array`, `numpy.ndarray`) from Julia or Julia arrays from Python.
- Helpful wrappers: interpret Python sequences, dictionaries, arrays, dataframes and IO streams as their Julia couterparts, and vice versa.
- Beautiful stack-traces.
- Works anywhere: tested on Windows, MacOS and Linux, 32- and 64-bit, Julia 1.0 upwards and Python 3.5 upwards.

This is actually two modules working together: a Julia one called `PythonCall` and a [tiny](https://github.com/cjdoris/PythonCall.jl/blob/master/juliacall/__init__.py) Python one called `juliacall`.
