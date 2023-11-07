# Benchmarks

## Benchmark 1: fill a dict

| Version | Factor | Time (microseconds) | Allocations |
| ------- | ------ | ------------------- | ----------- |
| Python | 1.0x | 280 | ? |
| PythonCall | 2.4x | 680 | 5008 |
| PythonCall + `unsafe_pydel!` | 1.1x | 300 | 1008 |
| PythonCall `@py` | 1.4x | 420 | 1002 |
| PythonCall `@py` + `@unsafe_pydel!` | 1.1x | 300 | 2 |
| PyCall | 5.4x | 1620 | 10987 |
| PyCall (readable but wrong) | 5.9x | 1784 | 11456 |

Python code:
```python-repl
>>> from timeit import timeit
>>> def test():
...     from random import random
...     x = {}
...     for i in range(1000):
...         x[str(i)] = i + random()
...     return x
...
>>> timeit("test()", N=1000, globals=globals())
```

PythonCall code:
```julia-repl
julia> using PythonCall, BenchmarkTools

julia> function test()
           random = pyimport("random").random
           x = pydict()
           for i in pyrange(1000)
               x[pystr(i)] = i + random()
           end
           return x
       end
test (generic function with 1 method)

julia> @benchmark test()
```

PythonCall + `unsafe_pydel!` code:
```julia-repl
julia> using PythonCall, BenchmarkTools

julia> function test()
           random = pyimport("random").random
           x = pydict()
           for i in pyrange(1000)
               k = pystr(i)
               r = random()
               v = i + r
               x[k] = v
               unsafe_pydel!(k)
               unsafe_pydel!(r)
               unsafe_pydel!(v)
               unsafe_pydel!(i)
           end
           return x
       end
test (generic function with 1 method)

julia> @benchmark test()
```

PythonCall `@py` code:
```julia-repl
julia> using PythonCall, BenchmarkTools

julia> test() = @py begin
           import random: random
           x = {}
           for i in range(1000)
               x[str(i)] = i + random()
               # Uncomment for unsafe_pydel! version:
               # @jl PythonCall.unsafe_pydel!(i)
           end
           x
       end
test (generic function with 1 method)

julia> @benchmark test()
```

PyCall code:
```julia-repl
julia> using PyCall, BenchmarkTools

julia> function test()
           random = pyimport("random")."random"
           x = pycall(pybuiltin("dict"), PyObject)
           str = pybuiltin("str")
           for i in pycall(pybuiltin("range"), PyObject, 1000)
               set!(x, pycall(str, PyObject, i), i + pycall(random, PyObject))
           end
           return x
       end
test (generic function with 1 method)

julia> @benchmark test()
```

PyCall (readable but wrong) code:
```julia-repl
julia> using PyCall, BenchmarkTools

julia> function test()
           random = pyimport("random").random
           x = pybuiltin("dict")()
           str = pybuiltin("str")
           for i in pybuiltin("range")(1000)
               x[str(i)] = i + random()
           end
           return x
       end
test (generic function with 1 method)

julia> @benchmark test()
```
