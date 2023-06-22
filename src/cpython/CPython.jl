"""
    module CPython

This module provides a direct interface to the Python C API.
"""
module C

import Base: @kwdef
import CondaPkg
import Pkg
using Libdl, Requires, UnsafePointers, Serialization, ..Utils

include("consts.jl")
include("pointers.jl")
include("extras.jl")
include("context.jl")
include("gil.jl")
include("jlwrap.jl")

function __init__()
    ccall(:jl_generating_output, Cint, ()) == 1 && return nothing

    init_context()
    with_gil() do
        init_jlwrap()
    end
end

end
