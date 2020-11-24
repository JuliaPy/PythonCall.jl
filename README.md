![Python.jl logo](https://raw.githubusercontent.com/cjdoris/Python.jl/master/logo-text.svg)
---

Bringing **Python** and **Julia** together for ultimate awesomeness.
- Simple syntax, just like regular Python.
- Intuitive and flexible conversions between Julia and Python: anything can be converted, you are in control.
- Fast non-copying conversion of numeric arrays in either direction: modify numpy arrays from Julia or Julia arrays from Python.
- Helpful wrappers: interpret Python sequences, dictionaries, arrays, dataframes and IO streams as their Julia couterparts.
- Beautiful stack-traces.

## Examples

Here we create a Julia array, interpret it as a NumPy array, modify the NumPy array and see the modification on the original.

```julia
julia> using Python

julia> np = pyimport("numpy")
py: <module 'numpy' from 'C:\\Users\\chris\\.julia\\conda\\3\\lib\\site-packages\\numpy\\__init__.py'>

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
py: <module 'pandas' from 'C:\\Users\\chris\\.julia\\conda\\3\\lib\\site-packages\\pandas\\__init__.py'>

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
