module PythonCall

const ROOT_DIR = dirname(@__DIR__)

include("API/API.jl")
include("Utils/Utils.jl")
include("C/C.jl")
include("GIL/GIL.jl")
include("GC/GC.jl")
include("Core/Core.jl")
include("Convert/Convert.jl")
include("PyMacro/PyMacro.jl")
include("Wrap/Wrap.jl")
include("JlWrap/JlWrap.jl")
include("Compat/Compat.jl")

# re-export everything
for m in [:Core, :Convert, :PyMacro, :Wrap, :JlWrap, :Compat]
    for k in names(@eval($m))
        if k != m
            @eval using .$m: $k
            @eval export $k
        end
    end
end

# non-exported API
using .Core: PyNULL, CONFIG

# not API but used in tests
for k in [
    :pyjlanytype,
    :pyjlarraytype,
    :pyjlvectortype,
    :pyjlbinaryiotype,
    :pyjltextiotype,
    :pyjldicttype,
    :pyjlmoduletype,
    :pyjlintegertype,
    :pyjlrationaltype,
    :pyjlrealtype,
    :pyjlcomplextype,
    :pyjlsettype,
    :pyjltypetype,
]
    @eval using .JlWrap: $k
end

end
