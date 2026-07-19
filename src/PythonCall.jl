module PythonCall

const ROOT_DIR = dirname(@__DIR__)

include("API/API.jl")
include("Utils/Utils.jl")
include("NumpyDates/NumpyDates.jl")
include("C/C.jl")
include("GIL/GIL.jl")
include("GC/GC.jl")
include("Core/Core.jl")
include("Convert/Convert.jl")
include("PyMacro/PyMacro.jl")
include("Wrap/Wrap.jl")
include("JlWrap/JlWrap.jl")
include("Compat/Compat.jl")

# not API but used in tests
for k in [
    :pyjlanytype,
    :pyjlarraytype,
    :pyjlvectortype,
    :pyjlbinaryiotype,
    :pyjltextiotype,
    :pyjldicttype,
    :pyjlsettype,
]
    @eval using .JlWrap: $k
end
function __init__()
    JlWrap.init_base_rule_high_priority()
    Wrap.init_wrap_rules_high_priority()
    Convert.init_numpy_high_priority()
    Convert.init_pyconvert_canonical()
end

end
