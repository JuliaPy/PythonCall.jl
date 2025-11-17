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
    :pyjlsettype,
]
    @eval using .JlWrap: $k
end

function __init__()
    Convert.init_pyconvert_extratypes()

    Convert.register_pyconvert_rules!()
    Convert.register_ctypes_rules!()
    Convert.register_numpy_rules!()
    Convert.register_pandas_rules!()
    Wrap.register_wrap_pyconvert_rules!()
    JlWrap.register_jlwrap_rules!()
    Convert.register_pyconvert_fallback_rules!()
end

end
