"""
    module CPython

This module provides a direct interface to the Python C API.
"""
module C

import Base: @kwdef
using Libdl, Requires, UnsafePointers, Serialization, Pkg, ..Utils, ..Conda

include("consts.jl")
include("pointers.jl")
include("extras.jl")
include("context.jl")
include("gil.jl")
include("jlwrap.jl")

function __init__()
    init_context()
    with_gil() do
        init_jlwrap()
    end
end

end
