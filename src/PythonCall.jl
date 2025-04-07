module PythonCall

include("API/API.jl")

module Internals

using ..PythonCall

Base.@kwdef mutable struct Config
    meta::String = ""
    auto_sys_last_traceback::Bool = true
    auto_fix_qt_plugin_path::Bool = true
end

include("Utils/Utils.jl")
include("C/C.jl")
include("GIL.jl")
include("GC.jl")
include("Core/Core.jl")
include("Convert/Convert.jl")
include("PyMacro.jl")
include("Wrap/Wrap.jl")
include("JlWrap/JlWrap.jl")
include("Compat/Compat.jl")

end

# config
"Configuration for PythonCall."
const CONFIG = Internals.Config()

# other consts
const PyNULL = Internals.C.PyNULL
const PYCONVERT_PRIORITY_WRAP = Internals.Convert.PYCONVERT_PRIORITY_WRAP
const PYCONVERT_PRIORITY_ARRAY = Internals.Convert.PYCONVERT_PRIORITY_ARRAY
const PYCONVERT_PRIORITY_CANONICAL = Internals.Convert.PYCONVERT_PRIORITY_CANONICAL
const PYCONVERT_PRIORITY_NORMAL = Internals.Convert.PYCONVERT_PRIORITY_NORMAL
const PYCONVERT_PRIORITY_FALLBACK = Internals.Convert.PYCONVERT_PRIORITY_FALLBACK

# # not API but used in tests
# for k in [
#     :pyjlanytype,
#     :pyjlarraytype,
#     :pyjlvectortype,
#     :pyjlbinaryiotype,
#     :pyjltextiotype,
#     :pyjldicttype,
#     :pyjlmoduletype,
#     :pyjlintegertype,
#     :pyjlrationaltype,
#     :pyjlrealtype,
#     :pyjlcomplextype,
#     :pyjlsettype,
#     :pyjltypetype,
# ]
#     @eval using .JlWrap: $k
# end

end
