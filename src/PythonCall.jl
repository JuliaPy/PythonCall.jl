module PythonCall

const VERSION = v"0.9.14"
const ROOT_DIR = dirname(@__DIR__)

include("CPython/_.jl")
const C = _CPython

include("Py/_.jl")
const GC = _Py.GC
const pynew = _Py.pynew

module _Compat
    using ..PythonCall
    using ..PythonCall: C, pynew

    # include("compat/gui.jl")
    # include("compat/ipython.jl")
    # include("compat/multimedia.jl")
    # include("compat/serialization.jl")
    # include("compat/tables.jl")

    function __init__()
        C.with_gil() do
            # init_gui()
            # init_pyshow()
            # init_tables()
        end
    end
end

# re-export everything
for m in [:_Py, :_Compat]
    for k in names(@eval($m))
        if k != m
            @eval using .$m: $k
            @eval export $k
        end
    end
end

end
