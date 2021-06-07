"""
    module CPython

This module provides a direct interface to the Python C API.
"""
module CPython

import Base: @kwdef
using Conda, Libdl

include("consts.jl")
include("pointers.jl")
include("context.jl")
include("properties.jl")
include("gil.jl")

end
