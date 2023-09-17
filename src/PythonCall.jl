module PythonCall

const VERSION = v"0.9.14"
const ROOT_DIR = dirname(@__DIR__)

include("utils/_.jl")

include("CPython/_.jl")
const C = _CPython

include("Py/_.jl")
for k in [:GC, :pynew, :pyisnull, :pycopy!, :getptr, :pydel!, :unsafe_pynext, :PyNULL]
    @eval const $k = _Py.$k
end

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

end
