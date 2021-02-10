![Python.jl logo](https://raw.githubusercontent.com/cjdoris/Python.jl/master/logo-text.svg)
---
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://cjdoris.github.io/Python.jl/stable)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://cjdoris.github.io/Python.jl/dev)
[![Test Status](https://github.com/cjdoris/Python.jl/workflows/Tests/badge.svg)](https://github.com/cjdoris/Python.jl/actions?query=workflow%3ATests)
[![Codecov](https://codecov.io/gh/cjdoris/Python.jl/branch/master/graph/badge.svg?token=A813UUIHGS)](https://codecov.io/gh/cjdoris/Python.jl)

Bringing **Python** and **Julia** together in seamless harmony:
- Call Python code from Julia and Julia code from Python via a symmetric interface.
- Simple syntax, so the Python code looks like Python and the Julia code looks like Julia.
- Intuitive and flexible conversions between Julia and Python: anything can be converted, you are in control.
- Fast non-copying conversion of numeric arrays in either direction: modify Python arrays (e.g. `bytes`, `array.array`, `numpy.ndarray`) from Julia or Julia arrays from Python.
- Helpful wrappers: interpret Python sequences, dictionaries, arrays, dataframes and IO streams as their Julia couterparts, and vice versa.
- Beautiful stack-traces.
- Works anywhere: tested on Windows, MacOS and Linux, 32- and 64-bit, Julia 1.0 upwards and Python 3.5 upwards.

⭐ If you like this, a GitHub star would be lovely thank you. ⭐

## Examples

Here we create a Julia array, interpret it as a NumPy array, modify the NumPy array and see the modification on the original.

```julia
julia> using Python

julia> np = pyimport("numpy")
py: <module 'numpy' from '...'>

julia> x = rand(2,3)
2×3 Array{Float64,2}:
 0.0100335  0.475726  0.54648
 0.718499   0.888354  0.821937

julia> y = np.asarray(x)
py:
array([[0.01003352, 0.47572603, 0.54648036],
       [0.71849857, 0.88835385, 0.82193677]])

julia> y[0,0] += 10
py: 10.010033515997105

julia> x
2×3 Array{Float64,2}:
 10.01      0.475726  0.54648
  0.718499  0.888354  0.821937
```

Here we create a Python list, interpret it as a Julia vector of strings, modify it and see the modification on the original.

```julia
julia> x = pylist(["apples", "oranges", "bananas"])
py: ['apples', 'oranges', 'bananas']

julia> y = PyList{String}(x)
3-element PyList{String}:
 "apples"
 "oranges"
 "bananas"

julia> push!(y, "grapes")
4-element PyList{String}:
 "apples"
 "oranges"
 "bananas"
 "grapes"

julia> x
py: ['apples', 'oranges', 'bananas', 'grapes']
```

Here we create a Pandas dataframe, interpret it as a Julia table satisfying the Tables.jl interface, and convert it to a DataFrames.jl dataframe.

```julia
julia> using DataFrames

julia> pd = pyimport("pandas")
py: <module 'pandas' from '...'>

julia> x = pd.DataFrame(pydict(x=[1,2,3], y=["a","b","c"], z=rand(3)))
py:
   x  y         z
0  1  a  0.159724
1  2  b  0.211601
2  3  c  0.629729

julia> y = DataFrame(PyPandasDataFrame(x))
3×4 DataFrame
│ Row │ index │ x     │ y        │ z        │
│     │ Int64 │ Int64 │ PyObject │ Float64  │
├─────┼───────┼───────┼──────────┼──────────┤
│ 1   │ 0     │ 1     │ py: 'a'  │ 0.159724 │
│ 2   │ 1     │ 2     │ py: 'b'  │ 0.211601 │
│ 3   │ 2     │ 3     │ py: 'c'  │ 0.629729 │

julia> y = DataFrame(PyPandasDataFrame(x, columntypes=[:y=>String]))
3×4 DataFrame
│ Row │ index │ x     │ y      │ z        │
│     │ Int64 │ Int64 │ String │ Float64  │
├─────┼───────┼───────┼────────┼──────────┤
│ 1   │ 0     │ 1     │ a      │ 0.159724 │
│ 2   │ 1     │ 2     │ b      │ 0.211601 │
│ 3   │ 2     │ 3     │ c      │ 0.629729 │
```
