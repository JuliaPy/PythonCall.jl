![Python.jl logo](https://raw.githubusercontent.com/cjdoris/Python.jl/master/logo-text.svg)
---

Bringing **Python** and **Julia** together for ultimate awesomeness.
- Simple syntax, just like regular Python.
- Intuitive and flexible conversions between Julia and Python: anything can be converted, you are in control.
- Fast non-copying conversion of numeric arrays in either direction: modify numpy arrays from Julia or Julia arrays from Python.
- Helpful wrappers: interpret Python sequences, dictionaries, arrays, dataframes and IO streams as their Julia couterparts.
- Beautiful stack-traces.

## Example

```julia
julia> using Python

julia> np = pyimport("numpy")
py: <module 'numpy' from 'C:\\Users\\chris\\.julia\\conda\\3\\lib\\site-packages\\numpy\\__init__.py'>

julia> xjl = rand(2,3)
2×3 Array{Float64,2}:
 0.0100335  0.475726  0.54648
 0.718499   0.888354  0.821937

julia> xpy = np.asarray(xjl)
py:
array([[0.01003352, 0.47572603, 0.54648036],
       [0.71849857, 0.88835385, 0.82193677]])

julia> xpy[0,0] += 10
py: 10.010033515997105

julia> xjl
2×3 Array{Float64,2}:
 10.01      0.475726  0.54648
  0.718499  0.888354  0.821937

julia> ypy = pylist(["apples", "oranges", "bananas"])
py: ['apples', 'oranges', 'bananas']

julia> yjl = PyList{String}(ypy)
3-element PyList{String}:
 "apples"
 "oranges"
 "bananas"

julia> push!(yjl, "grapes")
4-element PyList{String}:
 "apples"
 "oranges"
 "bananas"
 "grapes"

julia> ypy
py: ['apples', 'oranges', 'bananas', 'grapes']
```
