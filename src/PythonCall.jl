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

    for rule in Convert.pyconvert_rule_specs()
        pyconvert_add_rule(rule.func, rule.tname, rule.type, rule.scope)
    end
    for rule in Convert.ctypes_rule_specs()
        pyconvert_add_rule(rule.func, rule.tname, rule.type, rule.scope)
    end
    for rule in Convert.numpy_rule_specs()
        pyconvert_add_rule(rule.func, rule.tname, rule.type, rule.scope)
    end
    for rule in Convert.pandas_rule_specs()
        pyconvert_add_rule(rule.func, rule.tname, rule.type, rule.scope)
    end
    for rule in Wrap.wrap_pyconvert_rule_specs()
        pyconvert_add_rule(rule.func, rule.tname, rule.type, rule.scope)
    end
    for rule in JlWrap.jlwrap_rule_specs()
        pyconvert_add_rule(rule.func, rule.tname, rule.type, rule.scope)
    end
    for rule in Convert.pyconvert_fallback_rule_specs()
        pyconvert_add_rule(rule.func, rule.tname, rule.type, rule.scope)
    end
end

end
