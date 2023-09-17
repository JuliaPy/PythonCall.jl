module PythonCall

const VERSION = v"0.9.14"
const ROOT_DIR = dirname(@__DIR__)

include("utils/_.jl")
include("CPython/_.jl")
include("Py/_.jl")
include("pyconvert/_.jl")
include("pymacro/_.jl")
include("pywrap/_.jl")
include("compat/_.jl")

# re-export everything
for m in [:_Py, :_pyconvert, :_pymacro, :_pywrap, :_compat]
    for k in names(@eval($m))
        if k != m
            @eval using .$m: $k
            @eval export $k
        end
    end
end

# non-exported API
for k in [:C, :GC, :pynew, :pyisnull, :pycopy!, :getptr, :pydel!, :unsafe_pynext, :PyNULL]
    @eval const $k = _Py.$k
end

end
