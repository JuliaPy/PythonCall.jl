module PythonCall

include("api.jl")

module Internals

# include("Utils/Utils.jl")
include("C/C.jl")
include("GIL.jl")
include("GC.jl")
# include("Core/Core.jl")
# include("Convert/Convert.jl")
# include("PyMacro/PyMacro.jl")
# include("Wrap/Wrap.jl")
# include("JlWrap/JlWrap.jl")
# include("Compat/Compat.jl")

end

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
