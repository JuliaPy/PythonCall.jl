module PythonCall

const ROOT_DIR = dirname(@__DIR__)

"""
    PythonCall._is_embedded

Marks the running sysimage as embedded in a Python host. Set to `true` in a
PackageCompiler `script=` to bake the embedded path into the sysimage.
"""
const _is_embedded = Ref(false)

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
